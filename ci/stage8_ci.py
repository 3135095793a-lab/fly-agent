#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stage8_ci.py — 阶段8全部实验（CI版）

Part A2: NREM v2 三组对照（v1 / v2 / 纯随机）
Part B:  四能力（B1 decompose / B2 recombine / B3 associate / B4 novelty）
Part C:  C1 长期 sleep（v2, 40/100 轮）/ C2 iso 细扫 + σ20 / C3 软门控对照

路径：FLY_NET=out/mb2_W.npz；FLY_ROOT 默认脚本上两级。
用法：python3 ci/stage8_ci.py [--smoke]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('FLY_ROOT', os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, 't3_ci'))
from fly_agent import FlyAgent  # noqa: E402
from eval_kit import make_kwta_stepped, eval_noise_robustness  # noqa: E402

OUT = os.environ.get('S8_OUT', os.path.join(ROOT, 'results', 'agent_v2'))
os.makedirs(OUT, exist_ok=True)
RESULTS = {}


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


def gate_metrics(fly, keys, pats, kd=50):
    for k in keys[:kd]:
        fly.disconnect(k)
    bad = sum(1 for k, p in zip(keys[:kd], pats[:kd]) if (fly.recall(p) or {}).get('key') == k)
    ok = sum(1 for k, p in zip(keys[kd:], pats[kd:]) if (fly.recall(p) or {}).get('key') == k)
    for k in keys[:kd]:
        fly.reconnect(k)
    rec = sum(1 for k, p in zip(keys[:kd], pats[:kd]) if (fly.recall(p) or {}).get('key') == k)
    return bad, ok, rec


def stab_kwta(fly):
    enc = make_kwta_stepped(fly.ctx)
    r = eval_noise_robustness(fly.ctx, enc)
    return r['noise_0.05']['cos']


def setup_global(n=100):
    fly = FlyAgent(mode='global')
    rng = np.random.default_rng(20260918)
    pats = [fly.ctx.rand_pattern(rng) for _ in range(n)]
    for i, p in enumerate(pats):
        fly.remember(f'm{i}', p)
    rngq = np.random.default_rng(555)
    noisy = [fly.ctx.scrambler(p, 0.3, rngq) for p in pats]
    return fly, pats, noisy


# ---------------- Part A2 ----------------
def exp_A2(rounds=20):
    log('=' * 30)
    log('PART A2: NREM v2 comparison (v1 / v2 / pure-random)')
    log('=' * 30)
    t0 = time.time()
    out = {}
    for tag in ['v1', 'v2', 'rand']:
        fly, pats, noisy = setup_global()
        keys = list(fly.memory.keys())
        if tag == 'v1':
            fly.sleep(rounds=rounds, lam=0.90, noise_sigma=2.0, struct_update=True)
        elif tag == 'v2':
            fly.sleep(rounds=rounds, lam=0.90, noise_sigma=2.0, struct_update=False)
        else:
            apply_and_recodes(fly, perturb(np.ones(len(fly.g), np.float32), 2.0, 7003))
        t1 = t1_global(fly, pats)
        t3 = t3_global(fly, pats, noisy)
        bad, ok, rec = gate_metrics(fly, keys, pats)
        st = stab_kwta(fly)
        out[tag] = {'t1': t1, 't3': t3, 'bad': bad, 'ok': ok, 'rec': rec, 'stab': st,
                    'g_mean': float(fly.g.mean())}
        log(f'[A2-{tag}] T1={t1:.3f} T3={t3:.4f} | T4 {bad}/50,{ok}/50 | T5 {rec}/50 | stab={st:.4f} | g_mean={fly.g.mean():.3f}  ({time.time()-t0:.0f}s)')
    # 判据：v2 全部指标 ≥ v1 - 0.02
    v1, v2 = out['v1'], out['v2']
    checks = {
        't1': v2['t1'] >= v1['t1'] - 0.02,
        't3': v2['t3'] >= v1['t3'] - 0.02,
        't4_ok': v2['ok'] >= v1['ok'] - 1,
        't5': v2['rec'] >= v1['rec'] - 1,
        'stab': v2['stab'] >= v1['stab'] - 0.02,
    }
    out['verdict'] = {'checks': checks, 'pass': all(checks.values())}
    log(f'[A2] VERDICT: v2>=(v1-0.02) each: {checks} -> {"PASS (v2 定案)" if all(checks.values()) else "CHECK"}')
    RESULTS['A2'] = out


# ---------------- Part B ----------------
def exp_B1():
    log('=' * 30)
    log('PART B1: decompose (multi-part memory)')
    log('=' * 30)
    t0 = time.time()
    fly = FlyAgent(mode='segmented')
    rng = np.random.default_rng(20260918)
    pats = [fly.ctx.rand_pattern(rng) for _ in range(25)]
    for i, p in enumerate(pats):
        fly.remember(f'mp{i}', p, parts=4)
    res = {}
    for frac in [0.25, 0.5, 0.75, 1.0]:
        nb = max(1, int(round(4 * frac)))
        hit = 0
        for i, p in enumerate(pats):
            blocks = [p[j * 100:(j + 1) * 100] for j in range(nb)]
            r = fly.recall_blocked(blocks, k=1)
            if r and r[0]['key'] == f'mp{i}':
                hit += 1
        res[frac] = hit
        log(f'[B1] {int(frac*100)}% blocks: {hit}/25 = {hit/25:.2f}  ({time.time()-t0:.0f}s)')
    res['pass_50'] = res[0.5] / 25 > 0.8
    log(f'[B1] VERDICT (50% > 0.8): {res["pass_50"]}')
    RESULTS['B1'] = {str(k): v for k, v in res.items()}


def exp_B2(pairs=10):
    log('=' * 30)
    log('PART B2: recombine')
    log('=' * 30)
    t0 = time.time()
    fly = FlyAgent(mode='segmented')
    rng = np.random.default_rng(777)
    c_hit = a_hit = b_hit = 0
    for t in range(pairs):
        a = fly.ctx.rand_pattern(rng)
        b = fly.ctx.rand_pattern(rng)
        fly.remember(f'A{t}', a)
        fly.remember(f'B{t}', b)
        fly.recombine(f'C{t}', f'A{t}', f'B{t}', 0.5)
    for t in range(pairs):
        cpat = fly.memory[f'C{t}']['pattern']
        apat = fly.memory[f'A{t}']['pattern']
        bpat = fly.memory[f'B{t}']['pattern']
        rc = fly.recall(cpat)
        ra = fly.recall(apat)
        rb = fly.recall(bpat)
        c_hit += (rc and rc['key'] == f'C{t}')
        a_hit += (ra and ra['key'] == f'A{t}')
        b_hit += (rb and rb['key'] == f'B{t}')
    res = {'C': c_hit / pairs, 'A': a_hit / pairs, 'B': b_hit / pairs}
    res['pass'] = (c_hit / pairs > 0.9) and (a_hit == pairs) and (b_hit == pairs)
    log(f'[B2] C hit={c_hit}/{pairs} A={a_hit}/{pairs} B={b_hit}/{pairs} ({time.time()-t0:.0f}s) -> {"PASS" if res["pass"] else "CHECK"}')
    RESULTS['B2'] = res


def exp_B3():
    log('=' * 30)
    log('PART B3: associate (partial cue)')
    log('=' * 30)
    t0 = time.time()
    fly = FlyAgent(mode='segmented')
    rng = np.random.default_rng(20260918)
    pats = [fly.ctx.rand_pattern(rng) for _ in range(100)]
    for i, p in enumerate(pats):
        fly.remember(f'as{i}', p)
    res = {}
    for frac in [0.5, 0.7]:
        cut = int(round(400 * frac))
        hit = 0
        for i, p in enumerate(pats):
            r = fly.recall(p[:cut])
            if r and r['key'] == f'as{i}':
                hit += 1
        res[frac] = hit / 100
        log(f'[B3] {int(frac*100)}% cue: {hit}/100 = {hit/100:.2f}  ({time.time()-t0:.0f}s)')
    res['pass'] = (res[0.5] > 0.8) and (res[0.7] > 0.95)
    log(f'[B3] VERDICT (50%>0.8, 70%>0.95): {res["pass"]}')
    RESULTS['B3'] = {str(k): v for k, v in res.items()}


def exp_B4():
    log('=' * 30)
    log('PART B4: novelty')
    log('=' * 30)
    t0 = time.time()
    fly = FlyAgent(mode='segmented')
    rng = np.random.default_rng(20260918)
    pats = [fly.ctx.rand_pattern(rng) for _ in range(100)]
    for i, p in enumerate(pats):
        fly.remember(f'nv{i}', p)
    rngv = np.random.default_rng(606)
    in_sims = [fly.novelty(fly.ctx.scrambler(p, 0.1, rngv))[1] for p in pats]
    out_sims = [fly.novelty(fly.ctx.rand_pattern(rngv))[1] for _ in range(100)]
    thresh = (float(np.mean(in_sims)) + float(np.mean(out_sims))) / 2.0
    fly.novelty_thresh = thresh
    acc_in = float(np.mean([not fly.novelty(p)[0] for p in [fly.ctx.scrambler(q, 0.1, rngv) for q in pats[:50]]]))
    fresh = [fly.ctx.rand_pattern(rngv) for _ in range(50)]
    acc_out = float(np.mean([fly.novelty(q)[0] for q in fresh]))
    res = {'thresh': round(thresh, 4), 'in_mean': round(float(np.mean(in_sims)), 4),
           'out_mean': round(float(np.mean(out_sims)), 4), 'acc_in': acc_in, 'acc_out': acc_out}
    res['pass'] = (acc_in > 0.9) and (acc_out > 0.9)
    log(f'[B4] thresh={res["thresh"]} in={res["in_mean"]} out={res["out_mean"]} | acc_in={acc_in:.2f} acc_out={acc_out:.2f} ({time.time()-t0:.0f}s) -> {"PASS" if res["pass"] else "CHECK"}')
    RESULTS['B4'] = res


# ---------------- Part C ----------------
def exp_C1():
    log('=' * 30)
    log('PART C1: long-term sleep (v2, sigma=2, 40/100 rounds)')
    log('=' * 30)
    t0 = time.time()
    fly, pats, noisy = setup_global()
    out = {}
    for R in [40, 100]:
        fly.sleep(rounds=R if R == 40 else 60, lam=0.90, noise_sigma=2.0, struct_update=False)
        t1 = t1_global(fly, pats)
        t3 = t3_global(fly, pats, noisy)
        gm = float(fly.g.mean())
        traj = [round(x['g_mean'], 5) for x in fly.sleep_history[-1]['traj']]
        out[R] = {'g_mean': gm, 't1': t1, 't3': t3, 'traj_tail': traj[-5:]}
        log(f'[C1] after {R} rounds: g_mean={gm:.5f} T1={t1:.3f} T3={t3:.4f}  ({time.time()-t0:.0f}s)')
    RESULTS['C1'] = out


def exp_C2(rounds=20):
    log('=' * 30)
    log('PART C2: iso fine sweep + sigma20')
    log('=' * 30)
    t0 = time.time()
    # iso
    fly = FlyAgent(mode='segmented')
    rng = np.random.default_rng(20260918)
    pats = [fly.ctx.rand_pattern(rng) for _ in range(100)]
    keys = [f'm{i}' for i in range(100)]
    for k, p in zip(keys, pats):
        fly.remember(k, p)
    rngq = np.random.default_rng(555)
    noisy = [fly.ctx.scrambler(p, 0.3, rngq) for p in pats]
    sl = fly.sleep(rounds=rounds, lam=0.90, noise_sigma=0.0)
    gs = fly.g.copy()
    log(f'[C2] iso sleep done {sl}  ({time.time()-t0:.0f}s)')
    res = {}
    for s in [0.1, 0.25, 0.5, 2.0]:
        apply_and_recodes(fly, perturb(gs, s, 7003))
        hit = sum(1 for k, q in zip(keys, noisy) if (fly.recall(q) or {}).get('key') == k)
        res[s] = hit
        log(f'[C2] iso sigma={s}: T2 = {hit}/100  ({time.time()-t0:.0f}s)')
    # sigma=20 (global)
    fly2, pats2, noisy2 = setup_global()
    fly2.sleep(rounds=rounds, lam=0.90, noise_sigma=0.0)
    apply_and_recodes(fly2, perturb(fly2.g, 20.0, 7003))
    t3_20 = t3_global(fly2, pats2, noisy2)
    st_20 = stab_kwta(fly2)
    res['sigma20_t3'] = t3_20
    res['sigma20_stab'] = st_20
    log(f'[C2] global sigma=20: T3={t3_20:.4f} stab={st_20:.4f}  ({time.time()-t0:.0f}s)')
    RESULTS['C2'] = {str(k): v for k, v in res.items()}


def exp_C3():
    log('=' * 30)
    log('PART C3: soft vs hard gate')
    log('=' * 30)
    t0 = time.time()
    fly = FlyAgent(mode='segmented')
    rng = np.random.default_rng(303)
    pats = [fly.ctx.rand_pattern(rng) for _ in range(30)]
    keys = [f'g{i}' for i in range(30)]
    for k, p in zip(keys, pats):
        fly.remember(k, p)
    res = {}
    for mode in ['hard', 'soft']:
        for k in keys[:10]:
            fly.disconnect(k, mode=mode)
        hit = sum(1 for k, p in zip(keys[:10], pats[:10]) if (fly.recall(p) or {}).get('key') == k)
        ok = sum(1 for k, p in zip(keys[10:], pats[10:]) if (fly.recall(p) or {}).get('key') == k)
        for k in keys[:10]:
            fly.reconnect(k)
        res[mode] = {'disconnected_hit': hit, 'intact': ok}
        log(f'[C3] {mode}: disconnected-hit={hit}/10, intact={ok}/20  ({time.time()-t0:.0f}s)')
    RESULTS['C3'] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--only', type=str, default=None)
    args = ap.parse_args()
    rounds = 4 if args.smoke else 20

    log(f'STAGE8_CI START (smoke={args.smoke}, fly_root={ROOT})')
    t00 = time.time()
    if args.only in (None, 'A'):
        exp_A2(rounds)
    if args.only in (None, 'B'):
        exp_B1(); exp_B2(); exp_B3(); exp_B4()
    if args.only in (None, 'C'):
        exp_C1(); exp_C2(rounds); exp_C3()
    with open(os.path.join(OUT, 'stage8_results.json'), 'w') as f:
        json.dump(RESULTS, f, indent=1, ensure_ascii=False, default=str)
    log(f'STAGE8_CI DONE in {time.time()-t00:.0f}s')
    log('SUMMARY: ' + json.dumps(RESULTS, ensure_ascii=False, default=str)[:2500])


if __name__ == '__main__':
    main()