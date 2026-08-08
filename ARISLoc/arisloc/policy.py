#!/usr/bin/env python3
"""Actor-Critic policy network and action sampling utilities."""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from .config import ACTION_STOP_ID, NUM_DRL_ACTIONS, NUM_PHASES
from .traj import drl_action_allow_mask

def flat_to_multi(a: int) -> np.ndarray:
    return np.array([a // 2, a % 2], dtype=np.int64)

def multi_to_flat(act: np.ndarray) -> int:
    return int(act[0]) * 2 + int(act[1])

def drl_action_is_stop(action_id: int, num_phases: int=NUM_PHASES) -> bool:
    return int(action_id) >= int(num_phases)

class ActorCritic(nn.Module):

    def __init__(self, obs_dim: int, num_phases: int, hidden: int=128):
        super().__init__()
        self.num_phases = int(num_phases)
        self.num_actions = self.num_phases + 1
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.actor = nn.Linear(hidden, self.num_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return (self.actor(h), self.critic(h).squeeze(-1))

@torch.no_grad()
def policy_sample_action(policy: ActorCritic, obs_t: torch.Tensor, action_allow_mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, value = policy(obs_t)
    if action_allow_mask is not None:
        m = action_allow_mask.to(device=logits.device)
        logits = logits.masked_fill(m.unsqueeze(0) < 0.5, -1000000000.0)
    dist = Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return (action, log_prob, value)

def _legal_action_ids(num_actions: int, action_allow_mask: Optional[torch.Tensor]) -> List[int]:
    if action_allow_mask is None:
        return list(range(int(num_actions)))
    m = action_allow_mask.detach().cpu().numpy().ravel()
    n = min(int(num_actions), int(m.size))
    return [i for i in range(n) if float(m[i]) > 0.5]

def _behavior_log_prob(dist: Categorical, action: torch.Tensor, epsilon: float, n_legal: int) -> torch.Tensor:
    eps = float(epsilon)
    n = max(int(n_legal), 1)
    if eps <= 0.0:
        return dist.log_prob(action)
    pi_a = dist.log_prob(action).exp().clamp(min=1e-08)
    behavior_p = (1.0 - eps) * pi_a + eps / float(n)
    return torch.log(behavior_p.clamp(min=1e-08))

@torch.no_grad()
def policy_sample_action_epsilon_greedy(policy: ActorCritic, obs_t: torch.Tensor, action_allow_mask: Optional[torch.Tensor], epsilon: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, value = policy(obs_t)
    if action_allow_mask is not None:
        m = action_allow_mask.to(device=logits.device)
        logits = logits.masked_fill(m.unsqueeze(0) < 0.5, -1000000000.0)
    dist = Categorical(logits=logits)
    legal = _legal_action_ids(policy.num_actions, action_allow_mask)
    if not legal:
        legal = list(range(int(policy.num_actions)))
    n_legal = len(legal)
    eps = float(epsilon)
    if eps > 0.0 and np.random.random() < eps:
        action_id = int(np.random.choice(legal))
        action = torch.tensor(action_id, device=logits.device, dtype=torch.long)
    else:
        action = dist.sample()
    log_prob = _behavior_log_prob(dist, action, eps, n_legal)
    return (action, log_prob, value)

@torch.no_grad()
def policy_greedy_action(policy: ActorCritic, obs_t: torch.Tensor, action_allow_mask: Optional[torch.Tensor]=None) -> int:
    logits, _ = policy(obs_t)
    if action_allow_mask is not None:
        m = action_allow_mask.to(device=logits.device)
        logits = logits.masked_fill(m.unsqueeze(0) < 0.5, -1000000000.0)
    return int(logits.argmax(dim=-1).item())

def _seed_eval_action_rng(device: torch.device, seed: Optional[int]) -> None:
    if seed is None:
        return
    s = int(seed)
    torch.manual_seed(s)
    np.random.seed(s)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(s)

@torch.no_grad()
def policy_value(policy: ActorCritic, obs_t: torch.Tensor) -> float:
    _, value = policy(obs_t)
    return float(value.squeeze().cpu())
