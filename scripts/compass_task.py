#!/usr/bin/env python3
"""罗盘记忆任务：训练读出层解码「刚才的输入是A还是B」+ 抗噪测试"""
import os, numpy as np, time, json
from scipy.sparse import load_npz
OUT=os.environ.get('FLY_OUT', '/root/fly-agent/out')
W = load_npz(f'{OUT}/compass_W.npz').tocsr(); N = W.shape[0]
rng0 = np.random.default_rng(0); idx = rng0.permutation(N)
gA, gB = idx[:200], idx[200:400]
GAIN, ALPHA, TSTIM, HOLD = 1.0, 0.3, 5, 45

def trial(stim, noise=0.0, seed=0, amp=1.0):
    r = np.random.default_rng(seed)
    x = np.zeros(N, dtype=np.float32); x[gA if stim==0 else gB] = amp
    s = np.zeros(N, dtype=np.float32)
    for t in range(TSTIM+HOLD):
        d = GAIN*(W @ s) + (x if t < TSTIM else 0.0)
        if noise: d = d + noise*r.standard_normal(N).astype(np.float32)
        s = (1-ALPHA)*s + ALPHA*np.tanh(d)
    return s

t0=time.time(); NT=300
S = np.stack([trial(i%2, seed=200+i) for i in range(NT)])
Y = np.array([i%2 for i in range(NT)], dtype=np.float32)
print(f'数据 {S.shape}  {time.time()-t0:.1f}s  幅度|s|={np.abs(S).mean():.4f}', flush=True)

ntr=200
def ridge(X,Y,lam=1.0):
    Xb=np.hstack([X,np.ones((len(X),1),dtype=np.float32)])
    return np.linalg.solve(Xb.T@Xb+lam*np.eye(Xb.shape[1],dtype=np.float32), Xb.T@Y)
w = ridge(S[:ntr], Y[:ntr])
pred = (np.hstack([S[ntr:],np.ones((NT-ntr,1),dtype=np.float32)]) @ w) > 0.5
acc = (pred.astype(int)==Y[ntr:]).mean()
print(f'★ 干净准确率 {acc*100:.1f}%', flush=True)

res={'clean':round(float(acc),4)}
for nz in [0.1, 0.3, 0.5, 1.0]:
    Sn = np.stack([trial(i%2, noise=nz, seed=7000+i) for i in range(80)])
    p = (np.hstack([Sn,np.ones((80,1),dtype=np.float32)]) @ w) > 0.5
    a = (p.astype(int)==np.array([i%2 for i in range(80)])).mean()
    res[f'noise_{nz}']=round(float(a),4)
    print(f'   噪声 {nz}: {a*100:.1f}%', flush=True)
json.dump({'neurons':int(N),'nnz':int(W.nnz),'accuracy':res}, open(f'{OUT}/compass_task.json','w'), indent=2)
print('DONE', flush=True)
