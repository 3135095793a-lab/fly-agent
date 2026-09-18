#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NREM v1.1 —— 休眠态回放与巩固

设计：NREM-V1-DESIGN.md + NREM-DECISION.md（晴菜拍板 2026-09-17，Operit 审查融入）
实现要点：
- 休眠轨道：2400 步/轮（200 经历 × 12 步；块内前 5 步注入刺激；全程内在噪声 σ=0.05）
  —— A（自发底色）= 连续动力学 + 内在噪声；B（记忆浮现）= 每块注入一个经历刺激
- 两规则对照：
    A = 共激活计数   Δg = η_A × cnt / (P×T)
    B = 协方差（去均值乘积和） Δg = η_B × cov / (P×T)
  归一化口径 η_A = 7.2, η_B = 24.0
  （等价于未归一化口径 1.5e-5 / 5e-5，即每轮平均 Δh ≈ 2% / 1.4%；换算依据见 _probe_*）
- 更新：g += Δg → clip[0, g_max] → g *= λ → clip；g 截断（防炸）
- 同步：g 始终在副本上更新，轮末原子替换（半自动双轨：评估/推理读取已落盘的 g）
- 评估冻结 g：distance / retrieval（kwta_readout）+ stability（kwta_stepped）
- 扫参：规则{A,B} × λ{0.90,0.95,0.99,1.00} × 轮数{1,5,20} = 24 组

用法：
  python3 nrem_v1.py --smoke        # 冒烟（小库 32、1 轮、含一致性校验）
  python3 nrem_v1.py --baseline     # 基线评估（g=1）
  python3 nrem_v1.py --sweep        # 全量 24 组（自动跳过已完成）
  python3 nrem_v1.py --summary      # 生成汇总表
"""
import argparse
import json
import os
import sys
import time

import numpy as np

# 兼容 GitHub Actions 运行：允许利用 NREM_OUT / FLY_NET / FLY_PROTOCOL 环境变量
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_kit import (
    EvalContext, make_kwta_readout, make_kwta_stepped,
    eval_distance, eval_retrieval, eval_noise_robustness,
)

# 路径可配置（GitHub Actions 下用环境变量覆盖）
ROOT = os.environ.get('FLY_ROOT', '/sdcard/Download/fly-agent')
OUTDIR = os.environ.get('NREM_OUT', os.path.join(ROOT, 'results/dream_20260918_nrem_v1'))

CFG = {
    'n_exp': 200,
    'block_steps': 12,
    'stim_steps': 5,
    'sigma': 0.05,
    'eta_A': 7.2,
    'eta_B': 24.0,
    'g_max': 5.0,
    'g_min': 0.0,
    'exp_seed': 20260918,
    'lam_grid': [0.90, 0.95, 0.99, 1.00],
    'rounds_grid': [1, 5, 20],
    'rules': ['A', 'B'],
}


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def ensure_dirs():
    for sub in ['runs', 'g_traj', 'done']:
        os.makedirs(os.path.join(OUTDIR, sub), exist_ok=True)


# ---------------------------------------------------------------- 备份/加载
def load_ctx():
    ctx = EvalContext()
    ctx._pre = ctx.W.tocoo().row
    ctx._post = ctx.W.tocoo().col
    return ctx


def build_library(ctx, n, seed):
    rng = np.random.default_rng(seed)
    stims = [ctx.rand_pattern(rng) for _ in range(n)]
    x_cache = [ctx.stim_vec(s) for s in stims]
    return stims, x_cache


# ---------------------------------------------------------------- 单轮休眠
def sleep_round(ctx, s, x_cache, order, noise_rng, sigma, rule):
    """连续 2400 步（len(order)×12）；返回 (s_new, stat, T, diag)"""
    n_edges = ctx._W_base.nnz
    pre, post = ctx._pre, ctx._post
    steps = ctx.steps
    tstim = ctx.tstim
    T = len(order) * steps

    cnt = np.zeros(n_edges, np.int32)          # 规则 A 统计
    prod = np.zeros(n_edges, np.float64)       # 规则 B 用（Σ s_pre·s_post）
    sum_i = np.zeros(ctx.N, np.float64)        # 每神经元活动累计（求 μ）
    buf_pre = np.empty(n_edges, np.float32)
    buf_post = np.empty(n_edges, np.float32)

    steps_run = 0
    for exp_idx in order:
        x = x_cache[int(exp_idx)]
        for t in range(steps):
            xin = x if t < tstim else None
            s = ctx.sim_step(s, drive_x=xin, noise=sigma, rng=noise_rng)
            np.take(s, pre, out=buf_pre)
            np.take(s, post, out=buf_post)
            if rule == 'A':
                cnt += ((buf_pre > 0) & (buf_post > 0)).astype(np.int32)
            else:
                prod += buf_pre.astype(np.float64) * buf_post.astype(np.float64)
                sum_i += s
            steps_run += 1

    diag = {
        'T': T,
        'steps_run': steps_run,
        's_mean_abs': float(np.mean(np.abs(s))),
        's_frac_pos': float(np.mean(s > 0)),
        'mean_coactive_frac': (float(cnt.mean() / max(T, 1)) if rule == 'A' else None),
    }
    if rule == 'A':
        stat = cnt.astype(np.float64)
    else:
        mu = sum_i / max(T, 1)
        stat = prod - (T * mu[pre].astype(np.float64) * mu[post].astype(np.float64))
    return s, stat, T, diag


def compute_delta_g(stat, T, n_patterns, rule, cfg):
    eta = cfg['eta_A'] if rule == 'A' else cfg['eta_B']
    return (eta * stat / (n_patterns * T)).astype(np.float32)


def apply_update(g, dg, lam, cfg):
    g2 = np.clip(g + dg, cfg['g_min'], cfg['g_max'])
    g2 = np.clip(g2 * np.float32(lam), cfg['g_min'], cfg['g_max'])
    return g2.astype(np.float32)


def g_stats(g, dg):
    return {
        'mean': round(float(g.mean()), 6),
        'p50': round(float(np.percentile(g, 50)), 6),
        'p99': round(float(np.percentile(g, 99)), 6),
        'max': round(float(g.max()), 6),
        'min': round(float(g.min()), 6),
        'frac_lt_1e-3': round(float((g < 1e-3).mean()), 6),
        'frac_gt_4': round(float((g > 4).mean()), 6),
        'n_touched': int(np.count_nonzero(dg)),
        'mean_abs_dg': round(float(np.mean(np.abs(dg))), 8),
        'max_abs_dg': round(float(np.max(np.abs(dg))), 8),
    }


# ---------------------------------------------------------------- 评估
def eval_with_g(ctx, g, tag=''):
    t0 = time.time()
    ctx.apply_gain(g)  # 冻结：评估期间 g 不变
    enc_r = make_kwta_readout(ctx)
    enc_s = make_kwta_stepped(ctx)
    out = {
        'distance': eval_distance(ctx, enc_r),
        'retrieval': eval_retrieval(ctx, enc_r),
        'stability': eval_noise_robustness(ctx, enc_s),
        'eval_s': round(time.time() - t0, 1),
        'tag': tag,
    }
    return out


# ---------------------------------------------------------------- 单组运行
def group_tag(rule, lam, rounds):
    return f'{rule}_lam{lam:.2f}_r{rounds:02d}'


def run_group(ctx, x_cache, cfg, rule, lam, rounds, group_idx):
    tag = group_tag(rule, lam, rounds)
    done_path = os.path.join(OUTDIR, 'done', f'{tag}.json')
    if os.path.exists(done_path):
        log(f'[skip] {tag} (已完成)')
        return

    t0 = time.time()
    n_edges = ctx._W_base.nnz
    n_exp = len(x_cache)
    s = np.zeros(ctx.N, np.float32)
    g = np.ones(n_edges, np.float32)

    order_rng = np.random.default_rng(1000000 + group_idx * 10 + 1)
    noise_rng = np.random.default_rng(1000000 + group_idx * 10 + 2)

    traj = []
    for r in range(rounds):
        order = order_rng.permutation(n_exp)
        s, stat, T, diag = sleep_round(ctx, s, x_cache, order, noise_rng, cfg['sigma'], rule)
        dg = compute_delta_g(stat, T, n_exp, rule, cfg)
        if not np.all(np.isfinite(dg)):
            raise RuntimeError(f'{tag}: Δg 出现非有限值（爆炸），终止该组')
        g = apply_update(g, dg, lam, cfg)
        st = g_stats(g, dg)
        st['round'] = r + 1
        st['diag'] = diag
        traj.append(st)
        log(f'  [{tag}] round {r+1}/{rounds}: g_mean={st["mean"]:.5f} p99={st["p99"]:.4f} '
            f'max={st["max"]:.4f} |dg|={st["mean_abs_dg"]:.2e} touched={st["n_touched"]}')

    # g 轨迹落盘
    with open(os.path.join(OUTDIR, 'g_traj', f'{tag}.json'), 'w') as f:
        json.dump({'tag': tag, 'rule': rule, 'lam': lam, 'rounds': rounds, 'traj': traj}, f, indent=1)

    # 评估（冻结 g）
    eval_res = eval_with_g(ctx, g, tag)
    ctx.clear_gain()

    out = {
        'tag': tag, 'rule': rule, 'lam': lam, 'rounds': rounds,
        'eta': cfg['eta_A'] if rule == 'A' else cfg['eta_B'],
        'sigma': cfg['sigma'], 'n_exp': n_exp,
        'g_final': {'mean': float(g.mean()), 'p99': float(np.percentile(g, 99)), 'max': float(g.max()),
                    'min': float(g.min()), 'frac_lt_1e-3': float((g < 1e-3).mean())},
        'g_traj_last': traj[-1],
        'eval': eval_res,
        'elapsed_s': round(time.time() - t0, 1),
    }
    with open(os.path.join(OUTDIR, 'runs', f'{tag}.json'), 'w') as f:
        json.dump(out, f, indent=1)
    with open(done_path, 'w') as f:
        json.dump({'done': True, 'elapsed_s': out['elapsed_s']}, f)
    log(f'[done] {tag} in {out["elapsed_s"]}s')
    return out


# ---------------------------------------------------------------- 基线
def run_baseline(ctx):
    tag = 'baseline'
    done_path = os.path.join(OUTDIR, 'done', f'{tag}.json')
    if os.path.exists(done_path):
        log('[skip] baseline 已完成')
        return
    log('评估基线（g=1）...')
    ctx.clear_gain()
    res = eval_with_g(ctx, None, 'baseline')
    with open(os.path.join(OUTDIR, 'baseline.json'), 'w') as f:
        json.dump(res, f, indent=1)
    with open(done_path, 'w') as f:
        json.dump({'done': True}, f)
    log(f'[done] baseline: stab n05={res["stability"]["noise_0.05"]["cos"]} '
        f'rho={res["distance"]["rho_s"]} t1s@0.1={res["retrieval"]["noise_0.1"]["t1s"]}')


# ---------------------------------------------------------------- 冒烟
def smoke(ctx):
    log('== smoke：小库 32、1 轮（rule A & B）、含一致性校验 ==')
    # 1) sim_step 与 sim_base 一致性（g=1，12 步）
    stim = ctx.STAB_PATTERNS[0]
    x = ctx.stim_vec(stim)
    s1 = np.zeros(ctx.N, np.float32)
    for t in range(ctx.steps):
        xin = x if t < ctx.tstim else None
        s1 = ctx.sim_step(s1, drive_x=xin, noise=0.0)
    # 参考：独立复刻的单步循环（应与 sim_step 一致）
    s2 = np.zeros(ctx.N, np.float32)
    for t in range(ctx.steps):
        d = ctx.gain * (ctx._W_eff @ s2) + (x if t < ctx.tstim else 0.0)
        s2 = (1 - ctx.alpha) * s2 + ctx.alpha * np.tanh(d)
    diff = float(np.abs(s1 - s2).max())
    log(f'一致性校验: max|sim_step - 参考单步循环| = {diff:.2e} -> {"PASS" if diff < 1e-5 else "FAIL"}')

    # 2) 小库单轮（计时）
    stims, x_cache = build_library(ctx, 32, 7)
    for rule in ['A', 'B']:
        s = np.zeros(ctx.N, np.float32)
        rng = np.random.default_rng(1)
        t0 = time.time()
        order = np.random.default_rng(2).permutation(len(x_cache))
        s, stat, T, diag = sleep_round(ctx, s, x_cache, order, rng, CFG['sigma'], rule)
        dt = time.time() - t0
        dg = compute_delta_g(stat, T, len(x_cache), rule, CFG)
        log(f'rule {rule}: 1轮(32块/384步) {dt:.1f}s | stat mean={stat.mean():.4f} '
            f'|dg|=({np.abs(dg).mean():.2e}) diag={diag}')
    # 3) g 应用冒烟
    g = np.ones(ctx._W_base.nnz, np.float32)
    g[:100] = 0.5
    ctx.apply_gain(g)
    _ = ctx.sim_base(ctx.stim_vec(ctx.STAB_PATTERNS[1]))[ctx.KC][:5]
    ctx.clear_gain()
    log('apply_gain/clear_gain 冒烟通过')
    log('== smoke 完成 ==')


def _full_cfg():
    return CFG


def build_library_sig(ctx, n, seed):
    return build_library(ctx, n, seed)


# ---------------------------------------------------------------- 汇总
def summarize(ctx):
    runs_dir = os.path.join(OUTDIR, 'runs')
    files = sorted(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else []
    base = None
    bp = os.path.join(OUTDIR, 'baseline.json')
    if os.path.exists(bp):
        base = json.load(open(bp))
    rows = []
    stab = {}
    for fn in files:
        if not fn.endswith('.json'):
            continue
        d = json.load(open(os.path.join(runs_dir, fn)))
        tag = d['tag']
        ev = d['eval']
        st = ev['stability']['noise_0.05']
        row = {
            'tag': tag, 'rule': d['rule'], 'lam': d['lam'], 'rounds': d['rounds'],
            'g_mean': round(d['g_final']['mean'], 5), 'g_max': round(d['g_final']['max'], 4),
            'stab_n05': st['cos'], 'stab_min': st['cos_min'], 'stab_max': st['cos_max'],
            'rho_s': ev['distance']['rho_s'], 'rho_r': ev['distance']['rho_r'],
            'sparse': ev['distance']['sparse_abs'],
            't1s_01': ev['retrieval']['noise_0.1']['t1s'], 't1s_03': ev['retrieval']['noise_0.3']['t1s'],
            'elapsed_s': d['elapsed_s'],
        }
        if base:
            bs = base['stability']['noise_0.05']['cos']
            row['d_stab'] = round(st['cos'] - bs, 4)
            row['d_rho'] = round(ev['distance']['rho_s'] - base['distance']['rho_s'], 4)
            row['d_t1s'] = round(ev['retrieval']['noise_0.1']['t1s'] - base['retrieval']['noise_0.1']['t1s'], 4)
            row['lift_candidate'] = bool(row['d_stab'] > 0.05)
            row['no_regression'] = bool(row['d_rho'] >= -0.10 and row['d_t1s'] >= -0.10
                                        and 0.05 <= row['sparse'] <= 0.15)
        else:
            row.update({'d_stab': 'NA', 'd_rho': 'NA', 'd_t1s': 'NA',
                        'lift_candidate': 'NA', 'no_regression': 'NA'})
        rows.append(row)
        stab[tag] = ev['stability']
    # CSV
    if rows:
        import csv
        keys = list(rows[0].keys())
        with open(os.path.join(OUTDIR, 'sweep_table.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    # Markdown
    lines = ['# NREM v1.1 快筛结果表', '']
    if base:
        lines += [f"基线(g=1): stab_n05={base['stability']['noise_0.05']['cos']} \u00b1 "
                  f"[{base['stability']['noise_0.05']['cos_min']}, {base['stability']['noise_0.05']['cos_max']}] | "
                  f"rho_s={base['distance']['rho_s']} | t1s@0.1={base['retrieval']['noise_0.1']['t1s']} | "
                  f"sparse={base['distance']['sparse_abs']}", '']
    lines += ['| tag | rule | λ | rounds | g_mean | g_max | stab_n05 | Δstab | rho_s | Δrho | t1s@.1 | Δt1s | sparse | 上移苗头 | 不回退 |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append('| {tag} | {rule} | {lam} | {rounds} | {g_mean} | {g_max} | {stab_n05} | {d_stab} | '
                     '{rho_s} | {d_rho} | {t1s_01} | {d_t1s} | {sparse} | {lift_candidate} | {no_regression} |'.format(**r))
    with open(os.path.join(OUTDIR, 'sweep_table.md'), 'w') as f:
        f.write('\n'.join(lines))
    with open(os.path.join(OUTDIR, 'stability_intervals.json'), 'w') as f:
        json.dump({'baseline': base['stability'] if base else None, 'runs': stab}, f, indent=1)
    log(f'summary: {len(rows)} 组 -> sweep_table.md/csv, stability_intervals.json')


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--baseline', action='store_true')
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--group', type=str, default=None)
    args = ap.parse_args()

    ensure_dirs()
    t_all = time.time()

    if args.smoke:
        ctx = load_ctx()
        smoke(ctx)
        return

    if args.summary:
        summarize(None)
        return

    ctx = load_ctx()
    _, x_cache = build_library(ctx, CFG['n_exp'], CFG['exp_seed'])
    log(f'经历库: {len(x_cache)} 模式 | 网络 nnz={ctx._W_base.nnz}')

    if args.baseline:
        run_baseline(ctx)
        return

    if args.group:
        tag = args.group
        for rule in CFG['rules']:
            for lam in CFG['lam_grid']:
                for rounds in CFG['rounds_grid']:
                    if group_tag(rule, lam, rounds) == tag:
                        idx = (CFG['rules'].index(rule) * 12 + CFG['lam_grid'].index(lam) * 3
                               + CFG['rounds_grid'].index(rounds))
                        run_group(ctx, x_cache, CFG, rule, lam, rounds, idx)
                        return
        log(f'未找到组 {tag}')
        return

    if args.sweep:
        # 组索引统一为 rules×lam×rounds 枚举序（与 --group 一致）
        mapping = []
        for ri, rule in enumerate(CFG['rules']):
            for li, lam in enumerate(CFG['lam_grid']):
                for rj, rounds in enumerate(CFG['rounds_grid']):
                    gi = ri * 12 + li * 3 + rj
                    mapping.append((rule, lam, rounds, gi))
        # 执行顺序：先快（1 轮）→ 中（5）→ 慢（20）
        order = sorted(mapping, key=lambda x: (x[2], x[0], x[1]))
        log(f'全量扫参：{len(order)} 组')
        for rule, lam, rounds, gi in order:
            run_group(ctx, x_cache, CFG, rule, lam, rounds, gi)
        log('全部组完成，生成汇总...')
        summarize(ctx)
        with open(os.path.join(OUTDIR, 'DONE.txt'), 'w') as f:
            f.write(f'all done at {time.strftime("%Y-%m-%d %H:%M:%S")}, total {round(time.time()-t_all,1)}s\n')
        log(f'总耗时 {round(time.time()-t_all,1)}s')
        return

    ap.print_help()


if __name__ == '__main__':
    main()