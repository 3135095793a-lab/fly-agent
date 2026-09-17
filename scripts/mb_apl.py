#!/usr/bin/env python3
"""模拟 APL 全局抑制：g = g - inh * 全局活动水平"""
import os, numpy as np, time, json
from scipy.sparse import load_npz
OUT=os.environ.get('FLY_OUT', '/root/fly-agent/out')
W = load_npz(f'{OUT}/mb2_W.npz').tocsr(); N = W.shape[0]
din = np.asarray(W.getnnz(axis=0)).ravel()
KC = np.where(din >= np.percentile(din, 99))[0]
print(f'网络 {N:,} | KC候选 {len(KC):,}', flush=True)
rng = np.random.default_rng(3); pool = np.setdiff1d(np.arange(N), KC)
gA = rng.choice(pool, 400, replace=False); gB = rng.choice(np.setdiff1d(pool, gA), 400, replace=False)
ALPHA = 0.3

def run(gain, inh, amp=0.5, nin=5, steps=12, stim=0):
    x = np.zeros(N, dtype=np.float32); x[gA if stim==0 else gB] = amp
    s = np.zeros(N, dtype=np.float32)
    for t in range(steps):
        d = gain*(W @ s) + (x if t < nin else 0.0)
        if inh > 0: d = d - inh * s.mean()          # ← 全局抑制(APL)
        s = (1-ALPHA)*s + ALPHA*np.tanh(d)
    return s

print(f"{'gain':>5} {'inh':>6} | {'KC活跃':>8} {'全活跃':>8} {'A/B区分':>8} | 判定", flush=True)
print('-'*60, flush=True)
best=None
for gain in [0.5, 1.0, 2.0]:
    for inh in [0.0, 2.0, 5.0, 10.0, 20.0]:
        sa = run(gain, inh, stim=0); sb = run(gain, inh, stim=1)
        kc = float((np.abs(sa[KC])>0.05).mean()); al = float((np.abs(sa)>0.05).mean())
        diff = float(np.abs(sa-sb).mean())
        v = '★稀疏+区分' if (kc<0.3 and diff>0.02) else ('稀疏' if kc<0.3 else ('中' if kc<0.7 else '饱和'))
        if v.startswith('★') and best is None: best=(gain,inh,kc,diff)
        print(f'{gain:>5.1f} {inh:>6.1f} | {kc:>7.1%} {al:>7.1%} {diff:>8.4f} | {v}', flush=True)
print('DONE', flush=True)
if best: print(f'★ 最佳: gain={best[0]} inh={best[1]} KC活跃={best[2]:.1%} 区分={best[3]:.4f}', flush=True)
