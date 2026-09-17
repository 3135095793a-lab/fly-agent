#!/usr/bin/env python3
"""用训练后的增益跑 compass_task 评估（含 g=1 基线对照，同一脚本同一种子）"""
import os
import json
import time
import numpy as np
from scipy.sparse import load_npz, csr_matrix
import torch

OUT = os.environ.get('FLY_OUT', 'out')
W = load_npz(f'{OUT}/compass_W.npz').tocsr()
N = W.shape[0]
g = torch.load(f'{OUT}/compass_gain.pt', map_location='cpu').numpy().astype(np.float32)
Wc = W.tocoo()
g_at = g[Wc.row, Wc.col].astype(np.float32)
Wt = csr_matrix((Wc.data * g_at, (Wc.row, Wc.col)), shape=W.shape)
print(f'compass {N} | nnz {W.nnz} | gain applied', flush=True)


def eval_compass(Wx, tag):
    rng0 = np.random.default_rng(0)
    idx = rng0.permutation(N)
    gA, gB = idx[:200], idx[200:400]
    GAIN, ALPHA, TSTIM, HOLD = 1.0, 0.3, 5, 45

    def trial(stim, noise=0.0, seed=0, amp=1.0):
        r = np.random.default_rng(seed)
        x = np.zeros(N, dtype=np.float32)
        x[gA if stim == 0 else gB] = amp
        s = np.zeros(N, dtype=np.float32)
        for t in range(TSTIM + HOLD):
            d = GAIN * (Wx @ s) + (x if t < TSTIM else 0.0)
            if noise:
                d = d + noise * r.standard_normal(N).astype(np.float32)
            s = (1 - ALPHA) * s + ALPHA * np.tanh(d)
        return s

    t0 = time.time()
    NT = 300
    S = np.stack([trial(i % 2, seed=200 + i) for i in range(NT)])
    Y = np.array([i % 2 for i in range(NT)], dtype=np.float32)
    ntr = 200

    def ridge(X, Y, lam=1.0):
        Xb = np.hstack([X, np.ones((len(X), 1), dtype=np.float32)])
        return np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1], dtype=np.float32), Xb.T @ Y)

    w = ridge(S[:ntr], Y[:ntr])
    pred = (np.hstack([S[ntr:], np.ones((NT - ntr, 1), dtype=np.float32)]) @ w) > 0.5
    acc = (pred.astype(int) == Y[ntr:]).mean()
    res = {'clean': round(float(acc), 4)}
    for nz in [0.1, 0.3, 0.5, 1.0]:
        Sn = np.stack([trial(i % 2, noise=nz, seed=7000 + i) for i in range(80)])
        p = (np.hstack([Sn, np.ones((80, 1), dtype=np.float32)]) @ w) > 0.5
        a = (p.astype(int) == np.array([i % 2 for i in range(80)])).mean()
        res[f'noise_{nz}'] = round(float(a), 4)
    print(f'{tag}: clean {res["clean"]*100:.1f}% | n0.1 {res["noise_0.1"]*100:.1f}% | n0.3 {res["noise_0.3"]*100:.1f}% | n0.5 {res["noise_0.5"]*100:.1f}% | n1.0 {res["noise_1.0"]*100:.1f}%  ({time.time()-t0:.1f}s)', flush=True)
    return res


out = {'neurons': int(N), 'nnz': int(W.nnz)}
out['baseline_g1'] = eval_compass(W, 'baseline(g=1)')
out['trained'] = eval_compass(Wt, 'trained')
d = np.abs(g_at - 1.0)
out['gain_stats_used'] = {
    'sup_mean_abs_delta': round(float(d.mean()), 6),
    'sup_max_abs_delta': round(float(d.max()), 6),
    'frac_gt_0.01': round(float((d > 0.01).mean()), 4),
}
json.dump(out, open(f'{OUT}/eval_trained.json', 'w'), indent=2)
print('EVAL_SUMMARY ' + json.dumps(out), flush=True)