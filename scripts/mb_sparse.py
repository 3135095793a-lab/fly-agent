#!/usr/bin/env python3
"""测蘑菇体的稀疏编码：不同输入 → 稀疏且可区分的 KC 组合"""
import os, numpy as np, json, time
from scipy.sparse import load_npz
OUT=os.environ.get('FLY_OUT', '/root/fly-agent/out')
W = load_npz(f'{OUT}/mb_W.npz').tocsr(); N = W.shape[0]
din = np.asarray(W.getnnz(axis=0)).ravel()
# Kenyon 细胞候选 = 高扇入的神经元
kc_thr = np.percentile(din, 99)
KC = np.where(din >= kc_thr)[0]
print(f'网络 {N:,} | KC 候选(扇入≥{int(kc_thr)}): {len(KC):,}', flush=True)
rng = np.random.default_rng(3)
pool = np.setdiff1d(np.arange(N), KC)
gA = rng.choice(pool, 400, replace=False); gB = rng.choice(np.setdiff1d(pool, gA), 400, replace=False)

ALPHA = 0.3
def run(gain, stim, steps=12, amp=1.0):
    x = np.zeros(N, dtype=np.float32); x[gA if stim==0 else gB] = amp
    s = np.zeros(N, dtype=np.float32)
    for t in range(steps):
        d = gain*(W @ s) + x
        s = (1-ALPHA)*s + ALPHA*np.tanh(d)
    return s

print(f"{'gain':>6} | {'KC活跃率':>9} | {'全活跃率':>9} | {'A/B区分':>8} | {'KC区分':>8} | 判定", flush=True)
print('-'*68, flush=True)
for gain in [0.3, 0.5, 1.0, 2.0, 3.0]:
    sa = run(gain, 0); sb = run(gain, 1)
    kc_act = float((np.abs(sa[KC]) > 0.1).mean())
    all_act = float((np.abs(sa) > 0.1).mean())
    diff = float(np.abs(sa - sb).mean())
    kc_diff = float(np.abs(sa[KC] - sb[KC]).mean())
    v = '稀疏✓' if kc_act < 0.2 else ('中等' if kc_act < 0.5 else '饱和')
    print(f'{gain:>6.1f} | {kc_act:>8.1%} | {all_act:>8.1%} | {diff:>8.4f} | {kc_diff:>8.4f} | {v}', flush=True)
print('DONE', flush=True)
