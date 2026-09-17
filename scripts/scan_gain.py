#!/usr/bin/env python3
"""扫描 v2：加背景输入 + 电压下界，分段统计"""
import os, numpy as np
from scipy.sparse import load_npz
OUT=os.environ.get('FLY_OUT', '/root/fly-agent/out')
W = load_npz(f'{OUT}/cx_W.npz').tocsr(); N = W.shape[0]
tau, dt, Vth = 20.0, 1.0, 1.0
decay = float(np.exp(-dt/tau))
rng0 = np.random.default_rng(7); perm = rng0.permutation(N); grpA = perm[:300]

def run(gain, bg, amp=0.25, tstim=10, steps=60, seed=0, stim=True):
    r = np.random.default_rng(seed)
    v = np.zeros(N, dtype=np.float32); spk = np.zeros(N, dtype=np.float32)
    pre=post=0.0
    for t in range(steps):
        I = W @ spk
        v = v*decay + I*gain + bg
        v = np.maximum(v, -0.2)                       # 电压下界
        if stim and t < tstim: v[grpA] += amp
        fired = v >= Vth; spk = fired.astype(np.float32); v[fired] = 0.0
        nf = float(fired.sum())
        if t < tstim: pre += nf
        else: post += nf
    return pre/tstim/N, post/max(1,steps-tstim)/N

print(f"{'gain':>7} {'bg':>6} | {'刺激期':>8} {'刺激后':>8} | 判定")
print('-'*52)
for bg in [0.0, 0.005, 0.01, 0.02]:
    for g in [0.001, 0.002, 0.003, 0.004]:
        pr, po = run(g, bg)
        if pr*100 > 30: v='爆炸'
        elif po*100 > pr*100*0.7 and po*100>5: v='自持过强'
        elif pr*100 > 3: v='✓有响应'
        elif pr*100 > 0.5: v='弱响应'
        else: v='静默'
        print(f'{g:>7.4f} {bg:>6.3f} | {pr*100:>7.2f}% {po*100:>7.2f}% | {v}')
print('DONE')
