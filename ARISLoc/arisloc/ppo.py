#!/usr/bin/env python3
"""Rollout buffer, GAE advantage estimation, and PPO update step."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from .config import PPOConfig
from .policy import ActorCritic, _behavior_log_prob, multi_to_flat

@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor

class RolloutBuffer:

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self._obs_store = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.clear()

    def clear(self) -> None:
        self._n = 0
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.values: List[float] = []

    def __len__(self) -> int:
        return int(self._n)

    def obs_batch_numpy(self) -> np.ndarray:
        return self._obs_store[:self._n]

    def add(self, obs: np.ndarray, action_a: int, log_prob: float, reward: float, done: bool, value: float) -> None:
        i = int(self._n)
        if i >= self.capacity:
            raise RuntimeError('RolloutBuffer overflow')
        self._obs_store[i] = obs.astype(np.float32, copy=False)
        self._n = i + 1
        self.actions.append(int(action_a))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(1.0 if done else 0.0)
        self.values.append(float(value))

    def ready(self, min_steps: int) -> bool:
        return self._n >= int(min_steps)

def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_value: float, gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float64)
    last_gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_nonterminal = 1.0 - dones[t]
            next_value = last_value
        else:
            next_nonterminal = 1.0 - dones[t]
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    ret = adv + values
    return (adv.astype(np.float32), ret.astype(np.float32))

def ppo_update(policy: ActorCritic, optimizer: optim.Optimizer, batch: RolloutBatch, cfg: PPOConfig, device: torch.device, *, ent_coef: Optional[float]=None, ent_phase_coef: Optional[float]=None, ent_cont_coef: Optional[float]=None, ppo_epochs: Optional[int]=None) -> Tuple[float, float, float]:
    obs = batch.obs
    actions = batch.actions
    old_log_probs = batch.log_probs
    returns = batch.returns
    advantages = batch.advantages
    adv = advantages - advantages.mean()
    adv_std = adv.std()
    if adv_std > 1e-08:
        adv = adv / adv_std
    if ent_coef is not None:
        e_ent = float(ent_coef)
    else:
        e_ent = float(cfg.ent_phase_coef if ent_phase_coef is None else ent_phase_coef)
    n = obs.size(0)
    mb = min(int(cfg.ppo_minibatch), n)
    indices = np.arange(n)
    policy_loss_acc = 0.0
    value_loss_acc = 0.0
    ent_acc = 0.0
    n_mb = 0
    n_epochs = max(1, int(cfg.ppo_epochs if ppo_epochs is None else ppo_epochs))
    for _ in range(n_epochs):
        np.random.shuffle(indices)
        for start in range(0, n, mb):
            end = start + mb
            idx = indices[start:end]
            o = obs[idx]
            a = actions[idx]
            olp = old_log_probs[idx]
            ret = returns[idx]
            adv_b = adv[idx]
            logits, val = policy(o)
            if not torch.isfinite(logits).all() or not torch.isfinite(val).all():
                continue
            dist = Categorical(logits=logits)
            lp = dist.log_prob(a)
            ent = dist.entropy().mean()
            ratio = torch.exp(torch.clamp(lp - olp, -20.0, 20.0))
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1.0 - cfg.ppo_clip, 1.0 + cfg.ppo_clip) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(val, ret)
            loss = policy_loss + cfg.vf_coef * value_loss - e_ent * ent
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()
            policy_loss_acc += float(policy_loss.detach().cpu())
            value_loss_acc += float(value_loss.detach().cpu())
            ent_acc += float(ent.detach().cpu())
            n_mb += 1
    k = max(n_mb, 1)
    return (policy_loss_acc / k, value_loss_acc / k, ent_acc / k)
