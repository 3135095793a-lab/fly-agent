#!/usr/bin/env python3
"""phase4 补充分析（本地可复现）：MB 编码「共同成分」归因
- 复现 exp1 的全部 pair（同种子），比较：raw cos / Pearson（去每向量均值）/ pattern-centered（去全局均值模式）
- exp2 用 pattern-centered 相似度重算 top1/top5
- 随机无关对（floor）量化「相似度地板」
"""
import json
import numpy as np
from scipy.sparse import load_npz
from scipy.stats import spearmanr

cfg = json.load(open('ci/configs/mb2_sparse_best_meta.json'))
GAIN = float(cfg['config']['gain'])
KW = int(cfg['config']['mod']['kwta_k'])
STEPS = int(cfg['config']['steps'])
ALPHA = float(cfg['protocol']['alpha'])
TSTIM = int(cfg['protocol']['stim_steps'])
AMP = float(cfg['protocol']['amp'])

W = load_npz('out/mb2_W.npz').tocsr()
N = W.shape[0]
din = np.asarray(W.getnnz(axis=0)).ravel()
KC = np.where(din >= np.percentile(din, 99))[0]
pool = np.setdiff1d(np.arange(N), KC)


def kwta(vals):
    m = len(vals)
    k = max(1, int(np.ceil(m * KW / 100.0)))
    keep = np.argpartition(np.abs(vals), m - k)[m - k:]
    out = np.zeros(m, np.float32)
    out[keep] = vals[keep]
    return out


def mb(stim):
    x = np.zeros(N, np.float32)
    x[stim] = AMP
    s = np.zeros(N, np.float32)
    for t in range(STEPS):
        s = (1 - ALPHA) * s + ALPHA * np.tanh(GAIN * (W @ s) + (x if t < TSTIM else 0.0))
    return kwta(s[KC])


def rand_pattern(rng, k=400):
    return rng.choice(pool, k, replace=False)


def scrambler(base, frac, rng):
    n = int(round(len(base) * frac))
    if n <= 0:
        return base.copy()
    drop = rng.choice(base, n, replace=False)
    mask = np.ones(len(pool), dtype=bool)
    mask[np.searchsorted(pool, base)] = False
    rem = pool[mask]
    add = rng.choice(rem, n, replace=False)
    return np.concatenate([np.setdiff1d(base, drop), add])


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na * nb > 0 else 0.0


# ---- 复现 exp1 ----
rng = np.random.default_rng(42)
LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
pairs = []
for R in LEVELS:
    for k in range(40):
        base = rand_pattern(rng)
        part = scrambler(base, R, rng)
        pairs.append((R, base, part))

codes = {}
for R, base, part in pairs:
    codes.setdefault('b', []).append(mb(base))
    codes.setdefault('p', []).append(mb(part))
A = np.stack(codes['b'])
B = np.stack(codes['p'])
MU = A.mean(axis=0) + B.mean(axis=0)
MU /= 2.0

print('计算中...', flush=True)
raws, pears, patc = [], [], []
for i in range(len(pairs)):
    a, b = A[i], B[i]
    raws.append(cos(a, b))
    pears.append(float(np.corrcoef(a, b)[0, 1]))
    patc.append(cos(a - MU, b - MU))
raws = np.array(raws).reshape(5, 40)
pears = np.array(pears).reshape(5, 40)
patc = np.array(patc).reshape(5, 40)
xs = np.repeat([1.0 - R for R in LEVELS], 40)

res = {'per_level_means': {}}
for name, arr in [('raw_cos', raws), ('pearson', pears), ('pattern_centered_cos', patc)]:
    res['per_level_means'][name] = [round(float(v), 4) for v in arr.mean(axis=1)]
    rho = spearmanr(xs, arr.ravel())
    res[f'spearman_{name}'] = round(float(rho[0]), 4)
    print(f'{name}: per-level {res["per_level_means"][name]} | spearman {res[f"spearman_{name}"]}', flush=True)

# ---- floor：50 对随机无关 ----
rng3 = np.random.default_rng(99)
fl_raw, fl_pear, fl_pat = [], [], []
for _ in range(50):
    a = mb(rand_pattern(rng3))
    b = mb(rand_pattern(rng3))
    fl_raw.append(cos(a, b))
    fl_pear.append(float(np.corrcoef(a, b)[0, 1]))
    fl_pat.append(cos(a - MU, b - MU))
res['floor_random_pairs'] = {
    'raw_cos': round(float(np.mean(fl_raw)), 4),
    'pearson': round(float(np.mean(fl_pear)), 4),
    'pattern_centered_cos': round(float(np.mean(fl_pat)), 4),
}
print('floor(random pairs):', res['floor_random_pairs'], flush=True)

# ---- exp2 重算（pattern-centered）----
rng2 = np.random.default_rng(43)
bank = [rand_pattern(rng2) for _ in range(100)]
qsel = rng2.choice(100, 50, replace=False)
bank_codes = np.stack([mb(p) for p in bank])
MBANK = bank_codes.mean(axis=0)

exp2 = {}
for noise in [0.1, 0.3]:
    ranks = {'raw_cos': [], 'pattern_centered_cos': []}
    for bidx in qsel:
        qp = scrambler(bank[bidx], noise, rng2)
        qc = mb(qp)
        sims_raw = np.array([cos(qc, b) for b in bank_codes])
        sims_pat = np.array([cos(qc - MBANK, b - MBANK) for b in bank_codes])
        for metric, sims in [('raw_cos', sims_raw), ('pattern_centered_cos', sims_pat)]:
            order = np.argsort(-sims)
            ranks[metric].append(int(np.where(order == bidx)[0][0]))
    for metric in ['raw_cos', 'pattern_centered_cos']:
        r = np.array(ranks[metric])
        exp2[f'noise_{noise}_{metric}'] = {'top1': round(float((r == 0).mean()), 4), 'top5': round(float((r < 5).mean()), 4)}
        print(f'exp2 {noise} {metric}: top1 {exp2[f"noise_{noise}_{metric}"]["top1"]} top5 {exp2[f"noise_{noise}_{metric}"]["top5"]}', flush=True)

res['exp2_centered'] = exp2
json.dump(res, open('out/phase4_addendum.json', 'w'), indent=2, ensure_ascii=False)
print('ADDENDUM_DONE', flush=True)