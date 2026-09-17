#!/usr/bin/env python3
"""快速验证：降输入强度能否带来稀疏？"""
import os, numpy as np
from scipy.sparse import load_npz
OUT=os.environ.get('FLY_OUT', '/root/fly-agent/out')
W = load_npz(f'{OUT}/mb_W.npz').tocsr(); N = W.shape[0]
din = np.asarray(W.getnnz(axis=0)).ravel()
KC = np.where(din >= np.percentile(din, 99))[0]
rng = np.random.default_rng(3); pool = np.setdiff1d(np.arange(N), KC)
gA = rng.choice(pool, 400, replace=False)
ALPHA = 0.3

def run(gain, amp, nsteps_in, steps=12):
    x = np.zeros(N, dtype=np.float32); x[gA] = amp
    s = np.zeros(N, dtype=np.float32)
    for t in range(steps):
        d = gain*(W @ s) + (x if t < nsteps_in else 0.0)
        s = (1-ALPHA)*s + ALPHA*np.tanh(d)
    return s

print(f"{'gain':>5} {'amp':>5} {'in步':>4} | {'KC活跃':>8} {'全活跃':>8} | 判定", flush=True)
print('-'*52, flush=True)
for gain in [0.2, 0.5, 1.0]:
    for amp in [0.05, 0.2, 0.5]:
        for nin in [2, 5]:
            s = run(gain, amp, nin)
            kc = float((np.abs(s[KC])>0.05).mean()); al = float((np.abs(s)>0.05).mean())
            v = '✓稀疏' if kc < 0.25 else ('中' if kc < 0.6 else '饱和')
            print(f'{gain:>5.1f} {amp:>5.2f} {nin:>4d} | {kc:>7.1%} {al:>7.1%} | {v}', flush=True)
print('DONE', flush=True)
