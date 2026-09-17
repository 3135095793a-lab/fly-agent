#!/usr/bin/env python3
"""测试航向罗盘是否具有记忆：刺激 → 撤销 → 看状态能否自持且区分"""
import os, numpy as np, json
from scipy.sparse import load_npz
OUT=os.environ.get('FLY_OUT', '/root/fly-agent/out')
W = load_npz(f'{OUT}/compass_W.npz').tocsr(); N = W.shape[0]
print(f'罗盘: {N} 神经元 {W.nnz:,} 突触  抑制边 {(W.data<0).mean()*100:.1f}%', flush=True)
rng = np.random.default_rng(0)
idx = rng.permutation(N); gA, gB = idx[:200], idx[200:400]

def run(gain, alpha, stim, tstim=5, hold=45, amp=1.0):
    x = np.zeros(N, dtype=np.float32); x[gA if stim==0 else gB] = amp
    s = np.zeros(N, dtype=np.float32); snaps={}
    for t in range(tstim+hold):
        d = gain*(W @ s) + (x if t < tstim else 0.0)
        s = (1-alpha)*s + alpha*np.tanh(d)
        if t == tstim-1: snaps['stim'] = s.copy()
        if t == tstim+hold-1: snaps['end'] = s.copy()
    return snaps

print(f"{'gain':>6} {'alpha':>6} | {'刺激末|s|':>9} {'延迟末|s|':>9} | {'保持':>7} | {'A/B区分':>8} | 判定", flush=True)
print('-'*70, flush=True)
best=None
for gain in [0.5, 1.0, 1.5, 2.0, 3.0]:
    for alpha in [0.1, 0.3]:
        a = run(gain, alpha, 0); b = run(gain, alpha, 1)
        m1 = float(np.abs(a['stim']).mean()); m2 = float(np.abs(a['end']).mean())
        keep = m2/max(m1,1e-9)
        diff = float(np.abs(a['end'] - b['end']).mean())
        # 关键：延迟末的区分度（而不是刺激末）
        if m2 < 0.02: v='蒸发'
        elif keep > 0.6 and diff > 0.05: v='★ 有记忆'
        elif keep > 0.6: v='自持无区分'
        else: v='衰减'
        if v.startswith('★') and best is None: best=(gain,alpha,diff,keep)
        print(f'{gain:>6.1f} {alpha:>6.1f} | {m1:>8.4f} {m2:>8.4f} | {keep:>6.1%} | {diff:>8.4f} | {v}', flush=True)
print('DONE', flush=True)
if best: print(f'★ 最佳: gain={best[0]} alpha={best[1]} 区分={best[2]:.4f} 保持={best[3]:.1%}', flush=True)
