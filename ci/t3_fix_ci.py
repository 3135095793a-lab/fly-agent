#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""t3_fix_ci.py — 7.3b 全部剩余实验（GitHub Actions 版）

实验1：iso 修复诊断（sleep → 结构/幅度分离 → σ扫描）
实验2：σ 上界与机制（global，复用一次 sleep 的 g）
实验3：σ=2.0 多种子复核（复用实验2的 g）
实验4：ruleB 的 hybrid 口径评估

路径：FLY_NET=out/mb2_W.npz（在 workflow 里设置）；FLY_ROOT 默认脚本上两级。
用法：python3 ci/t3_fix_ci.py [--smoke]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))          # ci/
ROOT = os.environ.get('FLY_ROOT', os.path.dirname(HERE))   # repo root
sys.path.insert(0, os.path.join(HERE, 't3_ci'))
from fly_agent import FlyAgent  # noqa: E402
from eval_kit import make_kwta_stepped, eval_noise_robustness  # noqa: E402

OUT = os.environ.get('T3FIX_OUT', os.path.join(ROOT, 'results', 't3_fix'))
os.makedirs(OUT, exist_ok=True)


def log(msg):
    print(msg, flush=True)


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


def perturb(g, sigma, seed):
    if sigma <= 0:
        return g.astype(np.float32).copy()
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(len(g)).astype(np.float32)
    return np.clip(g * np.exp(sigma * z), 0.0, 5.0).astype(np.float32)


def apply_and_recodes(fly, g):
    fly.g = g.astype(np.float32).copy()
    fly._refresh_eff()
    if fly.mode == 'segmented':
        for kk, m in fly.memory.items():
            full = fly._encode_full(m['pattern'])
            fly.memory[kk]['code'] = full[m['slot']].copy()
    else:
        for kk in fly.memory:
            fly.memory[kk]['code'] = fly._encode(fly.memory[kk]['pattern'])


def t2_iso(fly, keys, noisy):
    return sum(1 for k, q in zip(keys, noisy) if (fly.recall(q) or {}).get('key') == k)


def t3_global(fly, pats, noisy):
    codes = [fly._encode(p) for p in pats]
    hit = 0
    for i in range(len(pats)):
        cq = fly._encode(noisy[i])
        if int(np.argmax([cos(cq, c) for c in codes])) == i:
            hit += 1
    return hit / len(pats)


def t1_global(fly, pats):
    codes = [fly._encode(p) for p in pats]
    hit = 0
    for i in range(len(pats)):
        if int(np.argmax([cos(codes[i], c) for c in codes])) == i:
            hit += 1
    return hit / len(pats)


def stab_kwta(fly):
    enc = make_kwta_stepped(fly.ctx)
    r = eval_noise_robustness(fly.ctx, enc)
    return r['noise_0.05']['cos']


def lib_cos(fly, pats):
    cn = np.stack([fly._encode(p) for p in pats]).astype(np.float64)
    cn = cn / (np.linalg.norm(cn, axis=1, keepdims=True) + 1e-9)
    gram = cn @ cn.T
    iu = np.triu_indices(gram.shape[0], 1)
    return float(np.mean(gram[iu]))


def setup_global():
    fly = FlyAgent(mode='global')
    rng = np.random.default_rng(20260918)
    pats = [fly.ctx.rand_pattern(rng) for _ in range(100)]
    keys = [f'm{i}' for i in range(100)]
    for k, p in zip(keys, pats):
        fly.remember(k, p)
    rngq = np.random.default_rng(555)
    noisy = [fly.ctx.scrambler(p, 0.3, rngq) for p in pats]
    return fly, pats, keys, noisy


def exp1_iso(rounds):
    log('=' * 30)
    log('EXPERIMENT 1: iso repair diagnostics')
    log('=' * 30)
    t0 = time.time()
    fly = FlyAgent(mode='segmented')
    rng = np.random.default_rng(20260918)
    pats = [fly.ctx.rand_pattern(rng) for _ in range(100)]
    keys = [f'm{i}' for i in range(100)]
    for k, p in zip(keys, pats):
        fly.remember(k, p)
    rngq = np.random.default_rng(555)
    noisy = [fly.ctx.scrambler(p, 0.3, rngq) for p in pats]

    base = t2_iso(fly, keys, noisy)
    log(f'[E1] iso no-sleep baseline: T2 = {base}/100  ({time.time()-t0:.0f}s)')

    sl = fly.sleep(rounds=rounds, lam=0.90, noise_sigma=0.0)
    log(f'[E1] sleep done: {sl}  ({time.time()-t0:.0f}s)')
    gs = fly.g.copy()

    variants = [
        ('orig', gs.copy()),
        ('shuffled', gs[np.random.default_rng(1234).permutation(len(gs))].copy()),
        ('uniform_mean', np.full(len(gs), float(gs.mean()), np.float32)),
    ]
    res_struct = {}
    for name, gv in variants:
        apply_and_recodes(fly, gv)
        r = t2_iso(fly, keys, noisy)
        res_struct[name] = r
        log(f'[E1] {name}: T2 = {r}/100  ({time.time()-t0:.0f}s)')

    res_sigma = {}
    best = None
    for sigma in [0.0, 0.25, 0.5, 1.0, 2.0]:
        gv = perturb(gs, sigma, 7003)
        apply_and_recodes(fly, gv)
        r = t2_iso(fly, keys, noisy)
        res_sigma[sigma] = r
        log(f'[E1] sigma={sigma}: T2 = {r}/100  ({time.time()-t0:.0f}s)')
        if r >= 99 and best is None:
            best = sigma
    log(f'[E1] RESULT: baseline={base}/100, best sigma (T2>=99) = {best}')
    with open(os.path.join(OUT, 'ci_exp1_iso.json'), 'w') as f:
        json.dump({'baseline': base, 'struct': res_struct, 'sigma': {str(k): v for k, v in res_sigma.items()}, 'best_sigma': best}, f)

    # E1b：iso 额外变体（no-sleep+σ / 睡后更高σ）
    log('[E1b] extra iso variants:')
    n_edges = len(gs)
    g1 = np.ones(n_edges, np.float32)
    extra = {}
    for name, gv in [('nosleep_s2', perturb(g1, 2.0, 7003)),
                     ('nosleep_s3', perturb(g1, 3.0, 7003)),
                     ('sleep_s3', perturb(gs, 3.0, 7003)),
                     ('sleep_s5', perturb(gs, 5.0, 7003))]:
        apply_and_recodes(fly, gv)
        r = t2_iso(fly, keys, noisy)
        extra[name] = r
        log(f'[E1b] {name}: T2 = {r}/100  ({time.time()-t0:.0f}s)')
    with open(os.path.join(OUT, 'ci_exp1b_iso_extra.json'), 'w') as f:
        json.dump(extra, f)


def exp2_global(rounds):
    log('=' * 30)
    log('EXPERIMENT 2: sigma upper bound & mechanism (global)')
    log('=' * 30)
    t0 = time.time()
    fly, pats, keys, noisy = setup_global()
    t3_pre = t3_global(fly, pats, noisy)
    log(f'[E2] pre-sleep T3 = {t3_pre:.4f}  ({time.time()-t0:.0f}s)')

    sl = fly.sleep(rounds=rounds, lam=0.90)
    log(f'[E2] sleep done: {sl}  ({time.time()-t0:.0f}s)')
    gs = fly.g.copy()
    n = len(gs)

    res_upper = {}
    for s in [2, 3, 5, 10]:
        gv = perturb(gs, s, 7003)
        apply_and_recodes(fly, gv)
        t1 = t1_global(fly, pats)
        t3 = t3_global(fly, pats, noisy)
        st = stab_kwta(fly)
        res_upper[s] = {'t1': t1, 't3': t3, 'stab': st}
        log(f'[E2] sigma={s}: T1={t1:.3f} T3={t3:.4f} stab_n05={st:.4f}  ({time.time()-t0:.0f}s)')

    log('[E2] controls (pure randomization, no sleep):')
    g1 = np.ones(n, np.float32)
    ctrl = {}
    for name, gv in [('g1+s2', perturb(g1, 2.0, 7003)),
                     ('g1+s3', perturb(g1, 3.0, 7003)),
                     ('uniform0.35+s2', perturb(np.full(n, float(gs.mean()), np.float32), 2.0, 7003))]:
        apply_and_recodes(fly, gv)
        r = t3_global(fly, pats, noisy)
        ctrl[name] = r
        log(f'[E2] control {name}: T3 = {r:.4f}  ({time.time()-t0:.0f}s)')

    log('[E2] lib_cos vs sigma:')
    res_lib = {}
    for s in [0, 0.5, 1, 2, 3, 5]:
        gv = perturb(gs, s, 7003)
        apply_and_recodes(fly, gv)
        lc = lib_cos(fly, pats)
        res_lib[s] = lc
        log(f'[E2] sigma={s}: lib_cos = {lc:.4f}  ({time.time()-t0:.0f}s)')

    with open(os.path.join(OUT, 'ci_exp2_global.json'), 'w') as f:
        json.dump({'t3_pre': t3_pre, 'upper': {str(k): v for k, v in res_upper.items()},
                   'controls': ctrl, 'lib_cos': {str(k): v for k, v in res_lib.items()}}, f)
    return fly, gs, pats, keys, noisy


def exp3_seeds(fly, gs, pats, noisy):
    log('=' * 30)
    log('EXPERIMENT 3: sigma=2.0 multi-seed')
    log('=' * 30)
    t0 = time.time()
    for seed in [777, 888, 999]:
        gv = perturb(gs, 2.0, seed)
        apply_and_recodes(fly, gv)
        r = t3_global(fly, pats, noisy)
        log(f'[E3] sigma=2.0 seed={seed}: T3 = {r:.4f}  ({time.time()-t0:.0f}s)')
    log('[E3] done')


def exp4_ruleB(rounds):
    log('=' * 30)
    log('EXPERIMENT 4: ruleB (hybrid readout)')
    log('=' * 30)
    t0 = time.time()
    fly, pats, keys, noisy = setup_global()
    sl = fly.sleep(rounds=rounds, lam=0.90, rule='B')
    log(f'[E4] sleepB done: {sl}  ({time.time()-t0:.0f}s)')

    ra = t3_global(fly, pats, noisy)
    log(f'[E4] ruleB no-perturb: T3 = {ra:.4f}  ({time.time()-t0:.0f}s)')

    gv = perturb(fly.g, 1.0, 7003)
    apply_and_recodes(fly, gv)
    rb = t3_global(fly, pats, noisy)
    log(f'[E4] ruleB +sigma1.0: T3 = {rb:.4f}  ({time.time()-t0:.0f}s)')

    with open(os.path.join(OUT, 'ci_exp4_ruleB.json'), 'w') as f:
        json.dump({'ruleB_no_perturb': ra, 'ruleB_sigma1': rb}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--only', type=str, default=None)
    args = ap.parse_args()
    rounds = 2 if args.smoke else 20

    log(f'T3FIX_CI START (rounds={rounds}, fly_root={ROOT})')
    t00 = time.time()
    if args.only in (None, '1'):
        exp1_iso(rounds)
    if args.only in (None, '2'):
        fly2, gs2, pats2, keys2, noisy2 = exp2_global(rounds)
        if args.only is None:
            exp3_seeds(fly2, gs2, pats2, noisy2)
    if args.only in (None, '4'):
        exp4_ruleB(rounds)
    log(f'T3FIX_CI DONE in {time.time()-t00:.0f}s')


if __name__ == '__main__':
    main()