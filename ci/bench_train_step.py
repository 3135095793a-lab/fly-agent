#!/usr/bin/env python3
"""训练单步基准：复刻 train_compass.py 的计算路径，实测 BENCH_STEPS 步耗时 + 环境信息"""
import os
import time
import json
import numpy as np
from scipy.sparse import load_npz
import torch

torch.manual_seed(0)
OUT = os.environ.get('FLY_OUT', 'out')
STEPS = int(os.environ.get('BENCH_STEPS', '10'))


def cpu_model():
    try:
        for line in open('/proc/cpuinfo'):
            if line.lower().startswith('model name'):
                return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return 'unknown'


env = {'nproc': os.cpu_count(), 'cpu': cpu_model()}
try:
    env['cgroup_cpu_max'] = open('/sys/fs/cgroup/cpu.max').read().strip()
except Exception:
    env['cgroup_cpu_max'] = None
try:
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemTotal'):
                env['mem_total'] = line.split(':', 1)[1].strip()
                break
except Exception:
    pass
env['torch'] = torch.__version__
env['torch_threads'] = torch.get_num_threads()
env['numpy'] = np.__version__

Wnp = load_npz(f'{OUT}/compass_W.npz')
N = Wnp.shape[0]
W0 = torch.tensor(Wnp.toarray(), dtype=torch.float32)
print(f'compass {N} neurons | dense {W0.numel() * 4 / 1e6:.0f} MB | torch {torch.__version__} | threads {torch.get_num_threads()}', flush=True)

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


# warmup：完整跑 1 步（不计时）
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
warmup_s = time.time() - t0
print(f'warmup 1 step: {warmup_s:.2f}s', flush=True)

# 评估耗时（train_compass 训练前后各跑一次这类评估）
t0 = time.time()
acc = readout_acc(W0 * g.detach())
eval_s = time.time() - t0
print(f'readout eval ({acc * 100:.1f}%): {eval_s:.2f}s', flush=True)

times, fwd_times, bwd_times = [], [], []
for step in range(STEPS):
    t0 = time.time()
    opt.zero_grad()
    _, sa = rollout(0)
    _, sb = rollout(1)
    _, sn = rollout(0, noise=0.15)
    t1 = time.time()
    sep = -torch.mean((sa - sb) ** 2)
    stab = torch.mean((sa - sn) ** 2)
    mag = (torch.mean(sa ** 2) - 0.25) ** 2
    loss = sep + 2.0 * stab + 1.0 * mag
    loss.backward()
    opt.step()
    t2 = time.time()
    times.append(t2 - t0)
    fwd_times.append(t1 - t0)
    bwd_times.append(t2 - t1)
    print(f'step {step + 1}/{STEPS}: {t2 - t0:.2f}s (forward {t1 - t0:.2f}s)', flush=True)

mean = sum(times) / len(times)
summary = {
    'runs': STEPS,
    'mean_step_s': round(mean, 3),
    'min_step_s': round(min(times), 3),
    'max_step_s': round(max(times), 3),
    'mean_forward_s': round(sum(fwd_times) / STEPS, 3),
    'mean_backward_s': round(sum(bwd_times) / STEPS, 3),
    'warmup_s': round(warmup_s, 3),
    'eval_readout_s': round(eval_s, 3),
    'projected_300_steps_min': round(mean * 300 / 60, 1),
    'projected_300_steps_plus_eval_min': round((mean * 300 + eval_s * 2) / 60, 1),
}
summary.update(env)
json.dump(summary, open(f'{OUT}/bench_train.json', 'w'), indent=2)
print('BENCH_SUMMARY ' + json.dumps(summary, ensure_ascii=False), flush=True)
