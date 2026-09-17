#!/usr/bin/env python3
"""线程数对比：同一训练步计算，2 线程 vs 4 线程"""
import os
import time
import json
import numpy as np
from scipy.sparse import load_npz
import torch

torch.manual_seed(0)
OUT = os.environ.get('FLY_OUT', 'out')
K_WARM = 2
K_TIME = int(os.environ.get('COMPARE_STEPS', '6'))

Wnp = load_npz(f'{OUT}/compass_W.npz')
N = Wnp.shape[0]
W0 = torch.tensor(Wnp.toarray(), dtype=torch.float32)
print(f'compass {N} | dense {W0.numel() * 4 / 1e6:.0f} MB | torch {torch.__version__}', flush=True)

ALPHA, TSTIM, HOLD = 0.3, 5, 25
T = TSTIM + HOLD


def run_config(threads, k_warm, k_time):
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    perm = torch.randperm(N)
    gA, gB = perm[:200], perm[200:400]
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

    def one_step():
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
        return float(loss)

    for _ in range(k_warm):
        one_step()
    times = []
    for _ in range(k_time):
        t0 = time.time()
        one_step()
        times.append(time.time() - t0)
    return times


results = {}
for th in [2, 4]:
    times = run_config(th, K_WARM, K_TIME)
    r = {'mean_s': round(sum(times) / len(times), 3), 'min_s': round(min(times), 3),
         'max_s': round(max(times), 3), 'all': [round(t, 3) for t in times]}
    results[f'threads_{th}'] = r
    print(f'threads={th}: mean {r["mean_s"]}s (min {r["min_s"]} / max {r["max_s"]})', flush=True)

results['speedup_4_vs_2'] = round(results['threads_2']['mean_s'] / results['threads_4']['mean_s'], 3)
results['note'] = f'warmup {K_WARM} steps per config, timed {K_TIME} steps'
json.dump(results, open(f'{OUT}/threads_compare.json', 'w'), indent=2)
print('COMPARE_SUMMARY ' + json.dumps(results), flush=True)