#!/usr/bin/env python3
"""实验1 v3：修正刺激参数（先验证发放，再跑任务）"""
import os, numpy as np, time, json
from scipy.sparse import load_npz

OUT = os.environ.get('FLY_OUT', '/root/fly-agent/out')
W = load_npz(f'{OUT}/cx_W.npz').tocsr()
N = W.shape[0]
rng = np.random.default_rng(7)
perm = rng.permutation(N)
grpA, grpB = perm[:300], perm[300:600]

tau, dt, Vth = 20.0, 1.0, 1.0
decay = float(np.exp(-dt/tau)); gain = 0.004
AMP, T_STIM, T_DELAY = 0.25, 10, 25
steps = T_STIM + T_DELAY

def run(stim, seed, amp=AMP, tstim=T_STIM, record=None, ret_tail=False):
    r = np.random.default_rng(seed)
    v = np.zeros(N, dtype=np.float32); spk = np.zeros(N, dtype=np.float32)
    tgt = grpA if stim == 0 else grpB
    act = np.zeros(N, dtype=np.float32); tail = []
    for t in range(steps):
        I = W @ spk; v = v*decay + I*gain
        if t < tstim: v[tgt] += amp
        fired = v >= Vth; spk = fired.astype(np.float32); v[fired] = 0.0
        act += spk
        if ret_tail and t >= steps-5: tail.append(spk[record].copy())
    if ret_tail: return np.mean(tail, axis=0)
    return act

t0=time.time(); aA = run(0, 0); aB = run(1, 0)
print(f'[探测] 刺激A激活 {(aA>0).sum():,} | 刺激B激活 {(aB>0).sum():,}  ({time.time()-t0:.1f}s)', flush=True)
if (aA>0).sum() < 50:
    print('!! 仍无发放，参数还需调', flush=True); raise SystemExit(1)
print('✓ 网络有响应，继续', flush=True)

score = np.maximum(aA, aB)
readout = np.argsort(score)[-500:]
print(f'读出层: 500 个（最活跃）', flush=True)

t0=time.time(); NT=240
X = np.array([run(i%2, 1000+i, record=readout, ret_tail=True) for i in range(NT)])
Y = np.array([i%2 for i in range(NT)], dtype=np.float32)
print(f'数据 {X.shape}  {time.time()-t0:.1f}s  非零 {(X>0).mean()*100:.1f}%', flush=True)

def ridge(X, Y, lam=1.0):
    Xb = np.hstack([X, np.ones((len(X),1),dtype=np.float32)])
    return np.linalg.solve(Xb.T@Xb + lam*np.eye(Xb.shape[1],dtype=np.float32), Xb.T@Y)
ntr = 160
w = ridge(X[:ntr], Y[:ntr])
pred = (np.hstack([X[ntr:], np.ones((NT-ntr,1),dtype=np.float32)]) @ w) > 0.5
acc = (pred.astype(int)==Y[ntr:]).mean()
base = max(Y[ntr:].mean(), 1-Y[ntr:].mean())
print(f'=== 准确率 {acc*100:.1f}%  (随机基线 {base*100:.1f}%) ===', flush=True)
json.dump({'neurons':int(N),'active_A':int((aA>0).sum()),'active_B':int((aB>0).sum()),
           'accuracy':round(float(acc),4),'baseline':round(float(base),4),
           'amp':AMP,'t_stim':T_STIM}, open(f'{OUT}/exp1_result.json','w'), indent=2)
print('DONE', flush=True)
