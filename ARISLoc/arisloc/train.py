#!/usr/bin/env python3
"""PPO training loop over multi-RP path-phase episodes."""

from __future__ import annotations
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple
import numpy as np
import torch
import torch.optim as optim
try:
    from tqdm import tqdm as _tqdm_bar
except ImportError:
    _tqdm_bar = None
from .config import *
from .caches import TrajCsvLRUCache, TrajFolderCatalog, _EpisodeCsvAppender
from .env import PathPhaseMultiRPEnv
from .localizer import LocalizerInferEngine, load_localizer, apply_drl_localizer_eval_config
from .plots import *
from .policy import *
from .ppo import *
from .traj import scan_traj_folders, step_reward

def _reward_after_sub_episode_loc(cfg: PPOConfig, engine: LocalizerInferEngine, *, obs: np.ndarray, action_a: int, log_prob: float, value: float, done: bool, rp_step_t: int, rss: np.ndarray, rp: str, traj_key: str, buf: RolloutBuffer, env: PathPhaseMultiRPEnv, ep_rp_errs: List[float], err_precomputed: Optional[float]=None) -> Tuple[float, np.ndarray]:
    if err_precomputed is not None and np.isfinite(err_precomputed):
        err = float(err_precomputed)
    else:
        err = _localize_error_m(engine, rss, rp, traj_key)
    r = float(step_reward(cfg, err, int(rp_step_t), True))
    buf.add(obs, int(action_a), log_prob, r, done, value)
    ep_rp_errs.append(err)
    if done:
        next_obs = env._rss_to_obs(env._last_rss) if env._last_rss.size > 0 else obs
    elif PER_STEP_LOCALIZE and env._last_rss.size > 0:
        if env._rp_step == 1 and env._path1_err_cache is not None and (0 <= int(env._path1_task_i) < len(env._path1_err_cache)):
            next_obs = env.obs_with_cached_path1_err()
        else:
            _, next_obs = _obs_after_step_localization(env, engine, env._last_rss, str(env._rp), str(env._parent_key or ''))
    else:
        next_obs = env._rss_to_obs(env._last_rss)
    return (r, next_obs)

def _episode_train_progress(total_episodes: int):
    n = int(total_episodes)
    if SHOW_TRAIN_PROGRESS and _tqdm_bar is not None:
        return _tqdm_bar(total=n, desc='PPO train', unit='episode', mininterval=0.25, dynamic_ncols=True, ascii=True)
    return None

def train_ppo(cfg: PPOConfig) -> None:
    t_init = time.perf_counter()
    print(f'[init] scan traj folders under {cfg.drl_data_root.resolve()} ...', flush=True)
    traj_index = scan_traj_folders(cfg)
    n_ok = sum((len(v) for v in traj_index.values()))
    print(f'[init] traj index: {n_ok} folders across {len(traj_index)} RPs ({time.perf_counter() - t_init:.1f}s)', flush=True)
    if n_ok == 0 and (not cfg.synthetic_rss):
        print('[warn] No traj folders; enabling --synthetic demo')
        cfg.synthetic_rss = True
    elif 'Test_Data' in cfg.drl_data_root.parts and cfg.enumerate_path1_train:
        print('[warn] drl_data_root points to Test_Data; prefer a DRL_RP tree.', flush=True)
    bundle = load_localizer(cfg)
    engine = LocalizerInferEngine(bundle, cfg)
    print('[init] building csv catalog ...', flush=True)
    catalog = TrajFolderCatalog(cfg.drl_data_root, traj_index)
    traj_cache: Optional[TrajCsvLRUCache] = None
    if cfg.traj_cache_enabled and cfg.traj_cache_max_gb > 0:
        max_b = int(float(cfg.traj_cache_max_gb) * 1024 ** 3)
        traj_cache = TrajCsvLRUCache(max_b)
    env = PathPhaseMultiRPEnv(cfg, traj_index, bundle, catalog, traj_cache)
    obs_dim = int(env.observation_space.shape[0])
    device = torch.device(cfg.device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f'[init] create PPO single-head policy on {device} ...', flush=True)
    policy = ActorCritic(obs_dim, cfg.num_phases).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)
    buf = RolloutBuffer(cfg.ppo_rollout_steps, obs_dim)
    train_obs_t = torch.empty((1, obs_dim), dtype=torch.float32, device=device)
    print(f'[init] policy ready ({time.perf_counter() - t_init:.1f}s since start)', flush=True)
    cache_msg = f'LRU csv cache ≤{cfg.traj_cache_max_gb}GB' if traj_cache is not None else 'csv cache off'
    n_path1_per_ep = env.count_path1_tasks_per_episode() if cfg.enumerate_path1_train and (not cfg.synthetic_rss) else 0
    ep_mode = f'path1 random csv x phase0-5/RP, sub-eps/ep <={n_path1_per_ep}' if cfg.enumerate_path1_train and (not cfg.synthetic_rss) else f'RPs/episode={env.n_rps}'
    print(f'[train] PPO | episodes={cfg.train_total_episodes} | rollout={cfg.ppo_rollout_steps} | {ep_mode} | {cache_msg} | reward=MADRL-CIL: exec=-η*t/T(η={cfg.reward_eta},T={cfg.reward_max_steps}) term=-err/{cfg.d_norm_m} | per-step loc→obs | loc_batch={cfg.loc_infer_batch_size} | path1_precompute={('on' if PRECOMPUTE_PATH1_LOCS else 'off')} | loc_cache={('on' if LOC_INFER_CACHE_ENABLED else 'off')} | eval J=err+{COMPOSITE_LAMBDA}·path | csv={('aligned-k' if SUB_EPISODE_ALIGNED_CSV else 'random')} | ent={('exp-anneal' if ENT_ANNEAL_ENABLED else 'fixed')} {(ENT_START if ENT_ANNEAL_ENABLED else cfg.ent_phase_coef)}→{cfg.ent_phase_coef} τ={entropy_anneal_decay_episodes(cfg.train_total_episodes):.0f} | actions={cfg.num_phases}+stop={cfg.num_phases + 1} | explicit_stop={('on' if cfg.require_explicit_stop else 'off')} stop≥depth{cfg.min_path_depth_for_stop} | ε-greedy={('on' if cfg.epsilon_greedy_enabled else 'off')} {cfg.epsilon_start:g}→{cfg.epsilon_min:g} τ={cfg.epsilon_decay:g} | lr={cfg.lr:g}→{LR_MIN:g} cosine@{LR_DECAY_START_EP}~{LR_DECAY_END_EP} | ppo_ep={PPO_EPOCHS}→{PPO_EPOCHS_LATE}@{PPO_EPOCHS_LATE_START_EP} | seg={cfg.seg_len_m}m', flush=True)
    reward_csv = Path(EPISODE_RETURNS_CSV)
    reward_csv.parent.mkdir(parents=True, exist_ok=True)
    phase_ep_csv = Path(PHASE_EPISODE_FRAC_CSV)
    phase_stats = PhaseActionStats(cfg.num_phases) if SAVE_PHASE_STATS else None
    eval_phase_stats: Optional[PhaseActionStats] = None
    stop_stats = StopDepthStats()
    ep_csv_header = ['episode', 'env_step_at_ep_end', 'ep_return', 'mean_reward_per_step', 'mean_loc_err_m', 'mean_path_len_m', 'composite_J', 'stop_d1_frac', 'stop_d2_frac', 'stop_d3_frac', 'stop_d4_frac', 'total_env_steps']
    if SAVE_PHASE_STATS:
        ep_csv_header.extend(_phase_frac_csv_header(cfg.num_phases))
    with open(reward_csv, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(ep_csv_header)
    reward_csv_app = _EpisodeCsvAppender(reward_csv)
    phase_ep_csv_app: Optional[_EpisodeCsvAppender] = None
    if SAVE_PHASE_STATS:
        with open(phase_ep_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['episode', *_phase_frac_csv_header(cfg.num_phases)])
        phase_ep_csv_app = _EpisodeCsvAppender(phase_ep_csv)
    episode_returns: List[float] = []
    episode_mean_rewards: List[float] = []
    episode_mean_err_m: List[float] = []
    episode_mean_path_len_m: List[float] = []
    episode_composite_j: List[float] = []
    print('[init] env.reset (build path1 phase0-5 random-csv queue for episode 1) ...', flush=True)
    t_reset = time.perf_counter()
    obs, _ = env.reset(seed=0)
    fill_path1_loc_cache(env, engine, cfg)
    if PER_STEP_LOCALIZE and env._path1_err_cache:
        obs = env.obs_with_cached_path1_err()
    if cfg.enumerate_path1_train and (not cfg.synthetic_rss):
        print(f'[init] episode 1 queue: {len(env._path1_tasks)} path1 sub-episodes (≤{n_path1_per_ep} = {env.n_rps} RP × {cfg.num_phases} phase) ({time.perf_counter() - t_reset:.1f}s)', flush=True)
    ep_return = 0.0
    ep_rp_errs: List[float] = []
    ep_rp_lens: List[float] = []
    ep_env_steps = 0
    global_step = 0
    num_updates = 0
    pl_sum = vl_sum = ent_sum = 0.0
    ep_bar = _episode_train_progress(cfg.train_total_episodes)
    if phase_stats is not None:
        phase_stats.begin_episode()
    stop_stats.begin_episode()

    def _obs_to_policy_tensor(obs_np: np.ndarray) -> torch.Tensor:
        o = np.asarray(obs_np, dtype=np.float32).ravel()
        if o.size != obs_dim:
            o = np.resize(o, obs_dim)
        train_obs_t[0].copy_(torch.from_numpy(o))
        return train_obs_t
    try:
        while len(episode_returns) < cfg.train_total_episodes:
            cur_ep = len(episode_returns) + 1
            eps_roll = epsilon_for_episode(cur_ep, cfg.train_total_episodes, cfg)
            obs_t = _obs_to_policy_tensor(obs)
            act_mask = drl_action_allow_mask(cfg, obs, device)
            with torch.no_grad():
                if cfg.epsilon_greedy_enabled and eps_roll > 0.0:
                    action_t, log_prob, value = policy_sample_action_epsilon_greedy(policy, obs_t, act_mask, eps_roll)
                else:
                    action_t, log_prob, value = policy_sample_action(policy, obs_t, act_mask)
            action_a = int(action_t.item())
            next_obs, _r_placeholder, term, trunc, info = env.step(action_a)
            if phase_stats is not None and (not bool(info.get('is_stop'))):
                phase_stats.record(int(info.get('phase_id', action_a)), int(info.get('path_depth', 0)))
            done = bool(term or trunc)
            rp_sub_done = bool(info.get('rp_sub_done'))
            rp_step_t = int(info.get('rp_step_t', 1))
            pt_obs = obs.astype(np.float32, copy=False)
            pt_log_prob = float(log_prob.squeeze().detach().cpu().item())
            pt_value = float(value.squeeze().detach().cpu().item())
            if rp_sub_done:
                term_rss = np.asarray(info.get('rss'), dtype=np.float32).ravel()
                term_rp = str(info.get('rp', ''))
                term_key = str(info.get('traj_key', ''))
                term_err: Optional[float] = None
                if PER_STEP_LOCALIZE:
                    term_err = _cached_or_localize_error_m(env, engine, term_rss, term_rp, term_key)
                stop_stats.record(int(info.get('path_depth', 0)))
                r, obs = _reward_after_sub_episode_loc(cfg, engine, obs=pt_obs, action_a=action_a, log_prob=pt_log_prob, value=pt_value, done=done, rp_step_t=rp_step_t, rss=term_rss, rp=term_rp, traj_key=term_key, buf=buf, env=env, ep_rp_errs=ep_rp_errs, err_precomputed=term_err)
                ep_return += float(r)
                fl = info.get('rp_path_len_m', float('nan'))
                if np.isfinite(fl):
                    ep_rp_lens.append(float(fl))
            else:
                if PER_STEP_LOCALIZE:
                    _, obs = _obs_after_step_localization(env, engine, np.asarray(info.get('rss'), dtype=np.float32).ravel(), str(info.get('rp', '')), str(info.get('traj_key', '')))
                else:
                    obs = next_obs
                r = float(step_reward(cfg, 0.0, rp_step_t, False))
                buf.add(pt_obs, action_a, pt_log_prob, r, done, pt_value)
                ep_return += r
            ep_env_steps += 1
            global_step += 1
            if buf.ready(cfg.ppo_rollout_steps):
                with torch.no_grad():
                    if done:
                        last_v = 0.0
                    else:
                        last_v = policy_value(policy, _obs_to_policy_tensor(obs))
                adv, ret = compute_gae(np.asarray(buf.rewards, dtype=np.float64), np.asarray(buf.values, dtype=np.float64), np.asarray(buf.dones, dtype=np.float64), last_v, cfg.gamma, cfg.gae_lambda)
                obs_np = np.nan_to_num(buf.obs_batch_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
                batch = RolloutBatch(obs=torch.from_numpy(obs_np).to(device, non_blocking=True), actions=torch.tensor(buf.actions, dtype=torch.long, device=device), log_probs=torch.tensor(buf.log_probs, dtype=torch.float32, device=device), returns=torch.tensor(ret, dtype=torch.float32, device=device), advantages=torch.tensor(adv, dtype=torch.float32, device=device), values=torch.tensor(buf.values, dtype=torch.float32, device=device))
                cur_ep = len(episode_returns) + 1
                cur_lr = lr_for_episode(cur_ep, cfg.train_total_episodes)
                for pg in optimizer.param_groups:
                    pg['lr'] = cur_lr
                cur_ppo_ep = ppo_epochs_for_episode(cur_ep)
                e_ent = entropy_coef_for_episode(cur_ep, cfg.train_total_episodes, ent_end=cfg.ent_phase_coef)
                pl, vl, ent = ppo_update(policy, optimizer, batch, cfg, device, ent_coef=e_ent, ppo_epochs=cur_ppo_ep)
                num_updates += 1
                pl_sum += pl
                vl_sum += vl
                ent_sum += ent
                buf.clear()
            if done:
                me = float(np.mean(ep_rp_errs)) if ep_rp_errs else float('nan')
                mpl = float(np.mean(ep_rp_lens)) if ep_rp_lens else float('nan')
                n_ep_steps = max(int(ep_env_steps), 1)
                mean_r = float(ep_return) / float(n_ep_steps)
                episode_returns.append(ep_return)
                episode_mean_rewards.append(mean_r)
                episode_mean_err_m.append(me)
                episode_mean_path_len_m.append(mpl)
                cj = composite_score_j(me, mpl, COMPOSITE_LAMBDA)
                episode_composite_j.append(cj)
                stop_fr = stop_stats.episode_fractions()
                ep_row: List[Any] = [len(episode_returns), global_step, ep_return, mean_r, me, mpl, cj, float(stop_fr[0]), float(stop_fr[1]), float(stop_fr[2]), float(stop_fr[3]), ep_env_steps]
                if phase_stats is not None:
                    ep_fr = phase_stats.episode_fractions()
                    ep_row.extend((float(ep_fr[p]) for p in range(cfg.num_phases)))
                reward_csv_app.writerow(ep_row)
                if phase_ep_csv_app is not None:
                    phase_ep_csv_app.writerow([len(episode_returns), *[float(ep_fr[p]) for p in range(cfg.num_phases)]])
                ep_return = 0.0
                ep_rp_errs = []
                ep_rp_lens = []
                ep_env_steps = 0
                obs, _ = env.reset(seed=global_step)
                fill_path1_loc_cache(env, engine, cfg)
                if PER_STEP_LOCALIZE and env._path1_err_cache:
                    obs = env.obs_with_cached_path1_err()
                if phase_stats is not None:
                    phase_stats.begin_episode()
                stop_stats.begin_episode()
                if ep_bar is not None:
                    ep_bar.update(1)
                    if num_updates > 0:
                        ep_bar.set_postfix_str(f'step={global_step} | upd={num_updates} | r={mean_r:.3f} | err={me:.2f}m | path={mpl:.1f}m | J={cj:.2f}', refresh=False)
            if global_step > 0 and global_step % cfg.log_every == 0 and (len(episode_mean_rewards) > 0):
                wn = min(int(REWARD_ROLL_EPISODES), len(episode_mean_rewards))
                cur_ep_log = len(episode_returns) + 1
                e_ent_log = entropy_coef_for_episode(cur_ep_log, cfg.train_total_episodes, ent_end=cfg.ent_phase_coef)
                eps_log = epsilon_for_episode(cur_ep_log, cfg.train_total_episodes, cfg)
                lr_log = lr_for_episode(cur_ep_log, cfg.train_total_episodes)
                ppo_ep_log = ppo_epochs_for_episode(cur_ep_log)
                msg = f'step {global_step}  ep={len(episode_mean_rewards)}/{cfg.train_total_episodes}  reward_ma{wn}={np.mean(episode_mean_rewards[-wn:]):.4f}  err_ma{wn}={np.mean(episode_mean_err_m[-wn:]):.3f}m  path_ma{wn}={np.mean(episode_mean_path_len_m[-wn:]):.2f}m  J_ma{wn}={np.nanmean(episode_composite_j[-wn:]):.2f}  ppo_upd={num_updates}  ent={e_ent_log:.4f}  ε={eps_log:.3f}  lr={lr_log:.2e}  ppo_ep={ppo_ep_log}'
                if stop_stats.total_stops() > 0:
                    sf = stop_stats.fractions()
                    msg += f'  |  stop d1={sf[0]:.0%} d2={sf[1]:.0%} d3={sf[2]:.0%} d4={sf[3]:.0%}'
                if phase_stats is not None and phase_stats.total_actions() > 0:
                    msg += f'  |  {_format_phase_frac_line(phase_stats)}'
                if ep_bar is not None:
                    ep_bar.write(msg)
                else:
                    print(msg)
    finally:
        reward_csv_app.close()
        if phase_ep_csv_app is not None:
            phase_ep_csv_app.close()
        if ep_bar is not None:
            ep_bar.close()
    if len(buf) > 0:
        with torch.no_grad():
            last_v = policy_value(policy, _obs_to_policy_tensor(obs))
        adv, ret = compute_gae(np.asarray(buf.rewards, dtype=np.float64), np.asarray(buf.values, dtype=np.float64), np.asarray(buf.dones, dtype=np.float64), last_v, cfg.gamma, cfg.gae_lambda)
        obs_np = np.nan_to_num(buf.obs_batch_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        batch = RolloutBatch(obs=torch.from_numpy(obs_np).to(device, non_blocking=True), actions=torch.tensor(buf.actions, dtype=torch.long, device=device), log_probs=torch.tensor(buf.log_probs, dtype=torch.float32, device=device), returns=torch.tensor(ret, dtype=torch.float32, device=device), advantages=torch.tensor(adv, dtype=torch.float32, device=device), values=torch.tensor(buf.values, dtype=torch.float32, device=device))
        final_ep = max(1, len(episode_returns))
        cur_lr = lr_for_episode(final_ep, cfg.train_total_episodes)
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr
        cur_ppo_ep = ppo_epochs_for_episode(final_ep)
        e_ent = entropy_coef_for_episode(final_ep, cfg.train_total_episodes, ent_end=cfg.ent_phase_coef)
        ppo_update(policy, optimizer, batch, cfg, device, ent_coef=e_ent, ppo_epochs=cur_ppo_ep)
        num_updates += 1
    out = Path(cfg.policy_save_path)
    torch.save(policy.state_dict(), out)
    print(f'[saved] {out}  (single-head ActorCritic, obs_dim={obs_dim}, actions={cfg.num_phases}+stop={cfg.num_phases + 1})')
    print(f'[saved] {reward_csv.resolve()}')
    if traj_cache is not None:
        print(f'[cache] csv hits={traj_cache.hits} misses={traj_cache.misses} entries={len(traj_cache._data)} ~{traj_cache._bytes / 1000000000.0:.2f}GB')
    if engine._result_cache is not None:
        rc = engine._result_cache
        tot = max(rc.hits + rc.misses, 1)
        print(f'[cache] loc infer hits={rc.hits} misses={rc.misses} hit_rate={rc.hits / tot:.1%} entries={len(rc._data)}')
    if SAVE_TRAIN_CURVE_PLOTS and len(episode_mean_rewards) > 1:
        try:
            sep: Optional[Tuple[Path, Path, Path]] = None
            if SAVE_SEPARATE_CURVE_PLOTS:
                sep = (Path(CURVE_REWARD_PNG), Path(CURVE_MEAN_ERR_PNG), Path(CURVE_PATH_LEN_PNG))
            save_training_curve_plots(episode_mean_rewards, episode_mean_err_m, episode_mean_path_len_m, int(CURVE_PLOT_MA_WINDOW), Path(CURVE_COMBINED_PNG), sep)
            print(f'[saved] {Path(CURVE_COMBINED_PNG).resolve()}  (mean reward per step / mean err / mean path m)')
        except Exception as ex:
            print(f'[info] skipped training curves: {ex}')
    if stop_stats.total_stops() > 0:
        try:
            sd_path = Path(STOP_DEPTH_STATS_CSV)
            save_stop_depth_stats_csv(sd_path, stop_stats, label='train')
            sf = stop_stats.fractions()
            print(f'[saved] {sd_path.resolve()}  (stop depth fractions: path2={sf[0]:.1%} path3={sf[1]:.1%} path4={sf[2]:.1%})')
        except Exception as ex:
            print(f'[info] skipped stop-depth stats: {ex}')
    if SAVE_PHASE_STATS and phase_stats is not None and (phase_stats.total_actions() > 0):
        try:
            stats_path = Path(PHASE_STATS_CSV)
            save_phase_stats_csv(stats_path, phase_stats, label='train')
            print(f'[saved] {stats_path.resolve()}  (phase counts & fractions, train)')
            print(f'[phase train] {_format_phase_frac_line(phase_stats)}')
            for d in phase_stats.depth_keys:
                if int(phase_stats.by_depth[d].sum()) > 0:
                    fr_d = phase_stats.depth_fractions(d)
                    parts = [f'p{p}={fr_d[p]:.1%}' for p in range(cfg.num_phases)]
                    print(f'[phase train path{d}]  ' + '  '.join(parts))
            plot_path = Path(CURVE_PHASE_DIST_PNG)
            if save_phase_distribution_plot(plot_path, phase_stats, title='PPO DRL phase selection (training)'):
                print(f'[saved] {plot_path.resolve()}')
        except Exception as ex:
            print(f'[info] skipped phase stats: {ex}')
    if SAVE_ERROR_CDF and (not cfg.synthetic_rss):
        try:
            eval_act = 'greedy argmax' if DRL_EVAL_DETERMINISTIC else f'stochastic (same as train, seed={EVAL_ACTION_SEED})'
            print(f'[eval] path1 csv × trained DRL → localization error CDF | action={eval_act} ...')
            eval_phase_stats = PhaseActionStats(cfg.num_phases) if SAVE_PHASE_STATS else None
            cdf_errs = evaluate_path1_csv_cdf(cfg, policy, env, engine, catalog, device, phase_stats=eval_phase_stats)
            cdf_path = Path(CURVE_ERROR_CDF_PNG)
            if cdf_errs and save_error_cdf_plot(cdf_errs, cdf_path, xlim_max_m=float(CDF_XLIM_MAX_M), label='PPO DRL (path1 csv × trained policy)'):
                print(f'[saved] {cdf_path.resolve()}  (path1-csv post-train CDF, n={len(cdf_errs)} csv samples)')
                try:
                    from cdf_export import save_cdf_bundle
                    save_cdf_bundle('drl_v2', cdf_errs, depths=[1] * len(cdf_errs), meta={'script': 'ARISLoc.py', 'eval_scope': 'path1 csv only'})
                except Exception as ex_cdf:
                    print(f'[warn] CDF export failed: {ex_cdf}')
            if eval_phase_stats is not None and eval_phase_stats.total_actions() > 0:
                eval_stats_path = Path(PHASE_STATS_CSV).with_name(Path(PHASE_STATS_CSV).stem + '_eval.csv')
                save_phase_stats_csv(eval_stats_path, eval_phase_stats, label='eval_cdf')
                print(f'[saved] {eval_stats_path.resolve()}  (phase stats, path1 csv eval)')
                print(f'[phase eval] {_format_phase_frac_line(eval_phase_stats)}')
                eval_plot = Path(CURVE_PHASE_DIST_PNG).with_name(Path(CURVE_PHASE_DIST_PNG).stem + '_eval.png')
                if save_phase_distribution_plot(eval_plot, eval_phase_stats, title='PPO DRL phase selection (path1 csv eval)'):
                    print(f'[saved] {eval_plot.resolve()}')
            if not cdf_errs:
                print('[warn] CDF eval has no valid samples')
        except Exception as ex:
            print(f'[info] skipped CDF plot: {ex}')
    elif SAVE_ERROR_CDF and cfg.synthetic_rss:
        print('[info] synthetic RSS mode skips path1 csv CDF eval')
