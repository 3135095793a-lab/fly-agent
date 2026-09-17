#!/usr/bin/env python3
"""换成 rate-based 模型（与 fly-self-driving 一致）：r = tanh(gain * W r)"""
import os, numpy as np, time, json
from scipy.sparse import load_npz
OUT=os.environ.get('FLY_OUT', '/root/fly-agent/out')
W = load_npz(f'{OUT}/cx_W.npz').tocsr(); N = W.shape[0]
print(f'网络 {N:,} 神经元 {W.nnz:,} 突触', flush=True)

def run(gain, alpha=0.3, steps=40, inp=None, seed=0, ret=False):
    r = np.zeros(N, dtype=np.float32)
    rng = np.random.default_rng(seed)
    x = np.zeros(N, dtype=np.float32) if inp is None else inp
    traj = []
    for t in range(steps):
        drive = gain * (W @ r) + x
        r = (1-alpha)*r + alpha*np.tanh(drive)
        traj.append(float(np.abs(r).mean()))
    return (np.array(traj), r) if ret else np.array(traj)

print('--- 稳定性扫描（无输入）---', flush=True)
print(f"{'gain':>8} | {'|r|均值':>9} | {'|r|max':>8} | 判定", flush=True)
for g in [0.2, 0.5, 1.0, 2.0, 4.0, 8.0]:
    _, r = run(g, ret=True)
    m, mx = float(np.abs(r).mean()), float(np.abs(r).max())
    v = '稳定' if m < 0.05 else ('活跃' if m < 0.4 else '饱和')
    print(f'{g:>8.2f} | {m:>8.4f} | {mx:>7.3f} | {v}', flush=True)

# 加输入测试
print(flush=True)
leg = np.random.default_rng(7).permutation(N); gA, gB = leg[:1500], leg[1500:3000]
for g in [0.5, 1.0, 2.0]:
    ia = np.zeros(N, dtype=np.float32); ia[gA] = 0.8
    ib = np.zeros(N, dtype=np.float32); ib[gB] = 0.8
    _, ra = run(g, inp=ia, ret=True); _, rb = run(g, inp=ib, ret=True)
    diff = np.abs(ra - rb)
    print(f'gain={g}: 输入A/B 响应差异 {diff.mean():.4f} | 最大 {diff.max():.3f} | 可区分 {"✓" if diff.mean()>0.01 else "✗"}', flush=True)
print('DONE', flush=True)
