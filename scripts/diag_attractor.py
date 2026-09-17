#!/usr/bin/env python3
"""诊断：网络饱和 vs 真记忆。扫描 GAIN，测「刺激撤掉后状态能否自持」"""
import os, numpy as np, json
from scipy.sparse import load_npz
OUT=os.environ.get('FLY_OUT', '/root/fly-agent/out')
W = load_npz(f'{OUT}/cx_W.npz').tocsr(); N = W.shape[0]
rng = np.random.default_rng(7); perm = rng.permutation(N)
gA, gB = perm[:1500], perm[1500:3000]
ALPHA, TSTIM = 0.3, 5

def run(gain, stim, hold=25):
    """stim 阶段 5 步，然后 hold 步不给输入"""
    x = np.zeros(N, dtype=np.float32); x[gA if stim==0 else gB] = 0.8
    s = np.zeros(N, dtype=np.float32)
    snap = None
    for t in range(TSTIM + hold):
        drive = gain*(W @ s) + (x if t < TSTIM else 0.0)
        s = (1-ALPHA)*s + ALPHA*np.tanh(drive)
        if t == TSTIM-1: snap = s.copy()      # 刺激结束瞬间
    return snap, s

print(f"{'gain':>6} | {'刺激末|r|':>10} | {'延迟末|r|':>10} | {'保持率':>8} | {'A/B区分':>8} | 判定")
print('-'*72)
for gain in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
    sa, ea = run(gain, 0); sb, eb = run(gain, 1)
    mag_end_stim = np.abs(sa).mean()
    mag_delay = np.abs(ea).mean()
    keep = mag_delay / max(mag_end_stim, 1e-9)
    diff = np.abs(ea - eb).mean()
    if mag_delay < 0.01: v = '状态蒸发'
    elif keep > 0.8 and diff > 0.05: v = '✓ 自持且有区分'
    elif keep > 0.8: v = '自持但无区分'
    else: v = '衰减'
    print(f'{gain:>6.2f} | {mag_end_stim:>9.4f} | {mag_delay:>9.4f} | {keep:>7.1%} | {diff:>8.4f} | {v}')
print('DONE')
