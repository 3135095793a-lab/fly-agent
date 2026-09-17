#!/usr/bin/env python3
"""实验2：rate 模型 + 延迟匹配任务（工作记忆）
   刺激5步 → 延迟15步 → 从网络状态读出"刚才看到的是A还是B" """
import os, numpy as np, time, json
from scipy.sparse import load_npz
OUT=os.environ.get('FLY_OUT', '/root/fly-agent/out')
W = load_npz(f'{OUT}/cx_W.npz').tocsr(); N = W.shape[0]
rng = np.random.default_rng(7); perm = rng.permutation(N)
gA, gB = perm[:1500], perm[1500:3000]

GAIN, ALPHA, TSTIM, TDELAY = 1.0, 0.3, 5, 15
TOTAL = TSTIM + TDELAY

def trial(stim, noise=0.0, seed=0):
    r = np.random.default_rng(seed)
    x = np.zeros(N, dtype=np.float32); x[gA if stim==0 else gB] = 0.8
    s = np.zeros(N, dtype=np.float32)
    for t in range(TOTAL):
        drive = GAIN*(W @ s) + (x if t < TSTIM else 0.0)
        if noise: drive = drive + noise*r.standard_normal(N).astype(np.float32)
        s = (1-ALPHA)*s + ALPHA*np.tanh(drive)
    return s

t0=time.time(); NT=240
S = np.stack([trial(i%2, seed=100+i) for i in range(NT)])
Y = np.array([i%2 for i in range(NT)], dtype=np.float32)
print(f'生成 {NT} trials  {time.time()-t0:.1f}s  状态|均值|={np.abs(S).mean():.4f}', flush=True)

# 特征选择：只用训练集选（避免泄漏）
ntr = 160
act = np.abs(S[:ntr]).mean(axis=0)
feat = np.argsort(act)[-2000:]
X = S[:, feat]
print(f'特征 {X.shape[1]} 维（按训练集活跃度选）', flush=True)

def ridge(X, Y, lam=10.0):
    Xb = np.hstack([X, np.ones((len(X),1),dtype=np.float32)])
    return np.linalg.solve(Xb.T@Xb + lam*np.eye(Xb.shape[1],dtype=np.float32), Xb.T@Y)
w = ridge(X[:ntr], Y[:ntr])
pred = (np.hstack([X[ntr:], np.ones((NT-ntr,1),dtype=np.float32)]) @ w) > 0.5
acc = (pred.astype(int)==Y[ntr:]).mean()
print(f'=== 干净条件 准确率 {acc*100:.1f}% ===', flush=True)

# 抗噪测试（这才是果蝇的强项）
res = {'clean': round(float(acc),4)}
for noise in [0.2, 0.5, 1.0]:
    Sn = np.stack([trial(i%2, noise=noise, seed=9000+i) for i in range(80)])
    Xn = Sn[:, feat]
    pn = (np.hstack([Xn, np.ones((80,1),dtype=np.float32)]) @ w) > 0.5
    an = (pn.astype(int)==np.array([i%2 for i in range(80)])).mean()
    res[f'noise_{noise}'] = round(float(an),4)
    print(f'噪声 {noise}: 准确率 {an*100:.1f}%', flush=True)
json.dump({'neurons':int(N),'feat':2000,'accuracy':res}, open(f'{OUT}/exp2_result.json','w'), indent=2)
print('DONE', flush=True)
