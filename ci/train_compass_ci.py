#!/usr/bin/env python3
"""CI 正式训练：复刻 train_compass.py 逻辑 + 指定线程数 + 每 CKPT_EVERY 步存 checkpoint（稀疏 delta）"""
import os
import time
import json
import numpy as np
from scipy.sparse import load_npz
import torch

torch.manual_seed(0)
THREADS = int(os.environ.get('TORCH_THREADS', '4'))
STEPS = int(os.environ.get('TRAIN_STEPS', '300'))
CKPT_EVERY = int(os.environ.get('CKPT_EVERY', '50'))
STEPS = max(1, min(STEPS, 5000))
torch.set_num_threads(THREADS)
OUT = os.environ.get('FLY_OUT', 'out')
os.makedirs(f'{OUT}/checkpoints', exist_ok=True)
print(f'torch {torch.__version__} | threads={torch.get_num_threads()} | steps={STEPS} | ckpt every {CKPT_EVERY}', flush=True)

Wnp = load_npz(f'{OUT}/compass_W.npz')
N = Wnp.shape[0]
W0 = torch.tensor(Wnp.toarray(), dtype=torch.float32)
Wc = Wnp.tocoo()
SUP_R = Wc.row.astype(np.int64)
SUP_C = Wc.col.astype(np.int64)
SUP_FLAT = torch.from_numpy(SUP_R * N + SUP_C)
print(f'compass {N} | nnz {Wnp.nnz} | dense {W0.numel() * 4 / 1e6:.0f} MB', flush=True)

perm = torch.randperm(N)
gA, gB = perm[:200], perm[200:400]
ALPHA, TSTIM, HOLD = 0.3, 5, 25
T = TSTIM + HOLD

g = torch.nn.Parameter(torch.ones_like(W0))
opt = torch.optim.Adam([g], lr=0.02)


def rollout(stim, noise=0.0, bs=1):
    x = torch.zeros(bs, N)
    x[:, gA if stim == 0 else gB] = 1.0
    s = torch.zeros(bs, N)
    for t in range(T):
        d = (W0 * g) @ s.t()
        d = d.t() + (x if t < TSTIM else 0)
        if noise:
            d = d + noise * torch.randn_like(d)
        s = (1 - ALPHA) * s + ALPHA * torch.tanh(d)
        if t == TSTIM - 1:
            s_stim = s
    return s_stim, s


def readout_acc(W_eff, trials=40):
    S, Y = [], []
    with torch.no_grad():
        for i in range(trials):
            _, s = rollout(i % 2)
            S.append(s.squeeze(0))
            Y.append(i % 2)
    S = torch.stack(S)
    Y = torch.tensor(Y, dtype=torch.float32)
    Xb = torch.cat([S, torch.ones(len(S), 1)], 1)
    w = torch.linalg.lstsq(Xb, Y.unsqueeze(1)).solution
    pred = (Xb @ w > 0.5).float().squeeze(1)
    return float((pred == Y).float().mean())


def gain_stats():
    flat = g.detach().view(-1)
    d = flat - 1.0
    supd = d[SUP_FLAT]
    d2 = d.clone()
    d2[SUP_FLAT] = 0.0
    return {
        'sup_mean_abs_delta': float(supd.abs().mean()),
        'sup_max_abs_delta': float(supd.abs().max()),
        'sup_frac_gt_0.01': float((supd.abs() > 0.01).float().mean()),
        'offsup_max_abs_delta': float(d2.abs().max()),
    }


def save_checkpoint(step):
    dsup = (g.detach().view(-1)[SUP_FLAT] - 1.0).cpu().numpy().astype(np.float32)
    np.savez_compressed(f'{OUT}/checkpoints/gain_delta_step{step:03d}.npz',
                        row=SUP_R, col=SUP_C, delta=dsup)
    meta = gain_stats()
    meta['step'] = step
    json.dump(meta, open(f'{OUT}/checkpoints/gain_step{step:03d}.json', 'w'), indent=2)
    print(f'[ckpt] step {step}: mean|d|={meta["sup_mean_abs_delta"]:.5f} max|d|={meta["sup_max_abs_delta"]:.4f} offsup={meta["offsup_max_abs_delta"]:.2e}', flush=True)
    return meta


t_all0 = time.time()
t0 = time.time()
acc0 = readout_acc(W0 * g.detach())
eval0_s = time.time() - t0
print(f'训练前（g=1，训练组）readout 准确率 {acc0*100:.1f}%  ({eval0_s:.0f}s)', flush=True)

history = []
step_times = []
checkpoints = []
for step in range(STEPS):
    t0 = time.time()
    opt.zero_grad()
    _, sa = rollout(0)
    _, sb = rollout(1)
    _, sn = rollout(0, noise=0.15)
    sep = -torch.mean((sa - sb) ** 2)
    stab = torch.mean((sa - sn) ** 2)
    mag = (torch.mean(sa ** 2) - 0.25) ** 2
    loss = sep + 2.0 * stab + 1.0 * mag
    loss.backward()
    opt.step()
    dt = time.time() - t0
    step_times.append(dt)
    history.append({'step': step + 1, 'loss': round(loss.item(), 5),
                    'sep': round(-sep.item(), 5), 'stab': round(stab.item(), 5),
                    't': round(dt, 3)})
    if step % 10 == 0 or step == STEPS - 1:
        print(f'step {step+1}/{STEPS}: loss={loss.item():.4f} sep={-sep.item():.4f} stab={stab.item():.4f} t={dt:.2f}s', flush=True)
    if (step + 1) % CKPT_EVERY == 0 or step == STEPS - 1:
        m = save_checkpoint(step + 1)
        m['loss'] = round(loss.item(), 5)
        checkpoints.append(m)

t0 = time.time()
acc1 = readout_acc(W0 * g.detach())
eval1_s = time.time() - t0
print(f'训练后（训练组）readout 准确率 {acc1*100:.1f}%  ({eval1_s:.0f}s)', flush=True)

torch.save(g.detach().cpu(), f'{OUT}/compass_gain.pt')
dsup_final = (g.detach().view(-1)[SUP_FLAT] - 1.0).cpu().numpy().astype(np.float32)
np.savez_compressed(f'{OUT}/compass_gain_delta.npz', row=SUP_R, col=SUP_C, delta=dsup_final)

final_stats = gain_stats()
summary = {
    'threads': int(torch.get_num_threads()),
    'steps': STEPS,
    'ckpt_every': CKPT_EVERY,
    'acc_before_train_group': round(acc0, 4),
    'acc_after_train_group': round(acc1, 4),
    'eval_before_s': round(eval0_s, 2),
    'eval_after_s': round(eval1_s, 2),
    'step_mean_s': round(sum(step_times) / len(step_times), 3),
    'step_min_s': round(min(step_times), 3),
    'step_max_s': round(max(step_times), 3),
    'total_train_loop_s': round(sum(step_times), 1),
    'total_elapsed_s': round(time.time() - t_all0, 1),
    'gain_final': final_stats,
    'checkpoints': checkpoints,
    'history': history,
}
json.dump(summary, open(f'{OUT}/train_compass_ci.json', 'w'), indent=2)
print('TRAIN_SUMMARY ' + json.dumps({k: v for k, v in summary.items() if k != 'history'}), flush=True)