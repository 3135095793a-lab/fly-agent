#!/usr/bin/env python3
"""扫描：兴奋增益 × 抑制强度倍数 —— 找临界态"""
import os, numpy as np
from scipy.sparse import load_npz, csr_matrix
OUT=os.environ.get('FLY_OUT', '/root/fly-agent/out')
W = load_npz(f'{OUT}/cx_W.npz').tocsr(); N = W.shape[0]
print(f'网络 {N:,} 神经元 {W.nnz:,} 突触', flush=True)
tau, dt, Vth = 20.0, 1.0, 1.0
decay = float(np.exp(-dt/tau))
rng0 = np.random.default_rng(7); perm = rng0.permutation(N); grpA = perm[:300]
is_inhib_edge = (W.data < 0)
print(f'抑制性边占比 {is_inhib_edge.mean()*100:.1f}%', flush=True)

def build(gain, inh_mult):
    d = W.data.copy()
    d[is_inhib_edge] *= inh_mult          # 抑制加权
    return csr_matrix((d, W.indices, W.indptr), shape=W.shape)

def run(Wm, gain, amp=0.3, tstim=10, steps=70, seed=0, stim=True):
    r = np.random.default_rng(seed)
    v = np.zeros(N, dtype=np.float32); spk = np.zeros(N, dtype=np.float32)
    pre=post=0.0
    for t in range(steps):
        I = Wm @ spk
        v = v*decay + I*gain
        v = np.maximum(v, -0.2)
        if stim and t < tstim: v[grpA] += amp
        fired = v >= Vth; spk = fired.astype(np.float32); v[fired] = 0.0
        nf = float(fired.sum())
        if t < tstim: pre += nf
        else: post += nf
    return pre/tstim/N*100, post/max(1,steps-tstim)/N*100

print(f"{'inh_x':>6} {'gain':>7} | {'刺激期%':>8} {'刺激后%':>8} | 判定", flush=True)
print('-'*54, flush=True)
best = None
for inh in [1.0, 2.0, 4.0, 8.0]:
    Wm = build(1.0, inh)
    for g in [0.002, 0.004, 0.008]:
        pr, po = run(Wm, g)
        if pr > 40: v='爆炸'
        elif 0.3 < pr < 40 and po < 5: v='✓ 有响应且衰减'
        elif po >= 5: v='自持'
        else: v='弱'
        if v.startswith('✓') and best is None: best = (inh, g, pr, po)
        print(f'{inh:>6.1f} {g:>7.4f} | {pr:>7.2f} {po:>7.2f} | {v}', flush=True)
print('DONE', flush=True)
if best: print(f'候选: inh_mult={best[0]}, gain={best[1]} → 刺激期 {best[2]:.2f}%, 刺激后 {best[3]:.2f}%', flush=True)
