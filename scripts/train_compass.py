#!/usr/bin/env python3
"""训练航向罗盘的突触增益（PyTorch）—— 目标：提升工作记忆抗噪性
   连接组结构冻结，只学习每个突触的 gain（对应 fly-self-driving 的做法）"""
import os, torch, numpy as np, json, time
from scipy.sparse import load_npz
torch.manual_seed(0)

OUT=os.environ.get('FLY_OUT', '/root/fly-agent/out')
Wnp = load_npz(f'{OUT}/compass_W.npz')
N = Wnp.shape[0]
W0 = torch.tensor(Wnp.toarray(), dtype=torch.float32)   # dense 4193^2 ≈ 70MB
print(f'罗盘 {N} 神经元 | dense {W0.numel()*4/1e6:.0f} MB', flush=True)

perm = torch.randperm(N)
gA, gB = perm[:200], perm[200:400]
ALPHA, TSTIM, HOLD = 0.3, 5, 25
T = TSTIM + HOLD

g = torch.nn.Parameter(torch.ones_like(W0))          # 每个突触的增益
opt = torch.optim.Adam([g], lr=0.02)

def rollout(stim, noise=0.0, bs=1):
    x = torch.zeros(bs, N); x[:, gA if stim==0 else gB] = 1.0
    s = torch.zeros(bs, N)
    for t in range(T):
        d = (W0 * g) @ s.t()                       # [N, bs]
        d = d.t() + (x if t < TSTIM else 0)
        if noise: d = d + noise*torch.randn_like(d)
        s = (1-ALPHA)*s + ALPHA*torch.tanh(d)
        if t == TSTIM-1: s_stim = s
    return s_stim, s

def readout_acc(W_eff, trials=40):
    """用简单线性读出评估（不入图）"""
    S, Y = [], []
    with torch.no_grad():
        for i in range(trials):
            _, s = rollout(i%2); S.append(s.squeeze(0)); Y.append(i%2)
    S = torch.stack(S); Y = torch.tensor(Y, dtype=torch.float32)
    Xb = torch.cat([S, torch.ones(len(S),1)], 1)
    w = torch.linalg.lstsq(Xb, Y.unsqueeze(1)).solution
    pred = (Xb @ w > 0.5).float().squeeze(1)
    return float((pred == Y).float().mean())

t0=time.time()
acc0 = readout_acc(W0*g.detach())
print(f'训练前 准确率 {acc0*100:.1f}%  ({time.time()-t0:.0f}s)', flush=True)

# 训练：让两个刺激产生更可分且更稳定的终态
for step in range(300):
    opt.zero_grad()
    _, sa = rollout(0); _, sb = rollout(1)
    _, sn = rollout(0, noise=0.15)
    # 损失1：不同刺激 → 不同终态（最大化区分）
    sep = -torch.mean((sa - sb)**2)
    # 损失2：有噪/无噪同刺激 → 相似终态（稳定性）
    stab = torch.mean((sa - sn)**2)
    # 损失3：保持幅度不要太小也不要饱和
    mag = (torch.mean(sa**2) - 0.25)**2
    loss = sep + 2.0*stab + 1.0*mag
    loss.backward(); opt.step()
    if step % 50 == 0:
        print(f'  step {step:3d} loss={loss.item():.4f} 区分={-sep.item():.4f} 稳定性={stab.item():.4f} ({time.time()-t0:.0f}s)', flush=True)

with torch.no_grad():
    acc1 = readout_acc(W0*g)
print(f'训练后 准确率 {acc1*100:.1f}%', flush=True)
gchg = float((g - 1).abs().mean())
print(f'增益平均变化 {gchg:.4f} | 最大 {float(g.abs().max()):.3f}', flush=True)
json.dump({'acc_before':round(acc0,4),'acc_after':round(acc1,4),
           'gain_change':round(gchg,4),'elapsed_s':round(time.time()-t0,1)},
          open(f'{OUT}/train_compass.json','w'), indent=2)
torch.save(g.detach(), f'{OUT}/compass_gain.pt')
print('DONE', flush=True)
