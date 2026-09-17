#!/usr/bin/env python3
"""第四阶段：相似判断（FlyHash 任务）
- 实验1 保距性：输入差异梯度 10~90% × 40 对 → 编码相似度（cosine + Jaccard）→ Spearman
- 实验2 检索：100 库模式 / 50 查询（加噪 0.1、0.3）→ top-1 / top-5 命中率
- 实验3 对照组：随机投影 + 同样 kWTA 稀疏化（FlyHash 风格），同样做 1 和 2
配置直接读取第三阶段 mb2_sparse_best_meta.json（不重新扫参）。结构冻结。
"""
import os
import time
import json
import numpy as np
from scipy.sparse import load_npz
from scipy.stats import spearmanr

OUT = os.environ.get('FLY_OUT', 'out')
FAST = os.environ.get('P4_FAST', '0') == '1'
CFG_PATH = os.environ.get('P4_CFG', 'ci/configs/mb2_sparse_best_meta.json')

cfg = json.load(open(CFG_PATH))
GAIN = float(cfg['config']['gain'])
KW_K = int(cfg['config']['mod']['kwta_k'])
STEPS = int(cfg['config'].get('steps', 12))
ALPHA = float(cfg['protocol']['alpha'])
TSTIM = int(cfg['protocol']['stim_steps'])
AMP = float(cfg['protocol']['amp'])
print(f'使用配置: gain={GAIN} k={KW_K} steps={STEPS} | alpha={ALPHA} stim={TSTIM} amp={AMP}', flush=True)

W = load_npz(f'{OUT}/mb2_W.npz').tocsr()
N = W.shape[0]
din = np.asarray(W.getnnz(axis=0)).ravel()
KC = np.where(din >= np.percentile(din, 99))[0]
pool = np.setdiff1d(np.arange(N), KC)
NK = len(KC)
print(f'网络 {N:,} | KC {NK} | 输入池 {len(pool):,}', flush=True)


def apply_kwta_kc(vals, kpct):
    m = len(vals)
    k = max(1, min(int(np.ceil(m * kpct / 100.0)), m))
    if k >= m:
        return vals.copy()
    keep = np.argpartition(np.abs(vals), m - k)[m - k:]
    out = np.zeros(m, dtype=np.float32)
    out[keep] = vals[keep]
    return out


def mb_encode(stim):
    x = np.zeros(N, dtype=np.float32)
    x[stim] = AMP
    s = np.zeros(N, dtype=np.float32)
    for t in range(STEPS):
        d = GAIN * (W @ s) + (x if t < TSTIM else 0.0)
        s = (1 - ALPHA) * s + ALPHA * np.tanh(d)
    return apply_kwta_kc(s[KC], KW_K)


_rp_rng = np.random.default_rng(7)
RP = _rp_rng.standard_normal((NK, N), dtype=np.float32)
RP /= np.sqrt(400.0)


def ctrl_encode(stim):
    x = np.zeros(N, dtype=np.float32)
    x[stim] = AMP
    return apply_kwta_kc(RP @ x, KW_K)


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def jac(a, b):
    sa = a != 0
    sb = b != 0
    u = (sa | sb).sum()
    return float((sa & sb).sum() / u) if u else 0.0


def rand_pattern(rng, k=400):
    return rng.choice(pool, k, replace=False)


def scrambler(base, frac, rng):
    """把 base 的 frac 比例神经元替换为新随机神经元（保持 400 个）"""
    n = int(round(len(base) * frac))
    if n <= 0:
        return base.copy()
    drop = rng.choice(base, n, replace=False)
    rem_mask = np.ones(len(pool), dtype=bool)
    rem_mask[np.searchsorted(pool, base)] = False
    rem = pool[rem_mask]
    add = rng.choice(rem, n, replace=False)
    keep = np.setdiff1d(base, drop)
    return np.concatenate([keep, add])


# ============ 实验 1：保距性 ============
LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
NP_P = 8 if FAST else 40
rng = np.random.default_rng(42)
inp_sim = np.zeros((len(LEVELS), NP_P), dtype=np.float32)
m_cos = np.zeros_like(inp_sim)
m_jac = np.zeros_like(inp_sim)
c_cos = np.zeros_like(inp_sim)
c_jac = np.zeros_like(inp_sim)
t0 = time.time()
for li, R in enumerate(LEVELS):
    for k in range(NP_P):
        base = rand_pattern(rng)
        part = scrambler(base, R, rng)
        ca, cb = mb_encode(base), mb_encode(part)
        za, zb = ctrl_encode(base), ctrl_encode(part)
        inp_sim[li, k] = 1.0 - R
        m_cos[li, k] = cos(ca, cb)
        m_jac[li, k] = jac(ca, cb)
        c_cos[li, k] = cos(za, zb)
        c_jac[li, k] = jac(za, zb)
    print(f'[exp1] 差异 {int(R*100)}%: MB cos {m_cos[li].mean():.3f} / jac {m_jac[li].mean():.3f} | 对照 cos {c_cos[li].mean():.3f} / jac {c_jac[li].mean():.3f}', flush=True)
print(f'exp1 用时 {time.time()-t0:.1f}s', flush=True)

x = inp_sim.ravel()
sp = {}
for name, arr in [('mb_cos', m_cos), ('mb_jac', m_jac), ('ctrl_cos', c_cos), ('ctrl_jac', c_jac)]:
    rho, p = spearmanr(x, arr.ravel())
    sp[name] = {'rho': round(float(rho), 4), 'p': float(p)}
print(f"[exp1] Spearman MB: cos {sp['mb_cos']['rho']} jac {sp['mb_jac']['rho']} | CTRL: cos {sp['ctrl_cos']['rho']} jac {sp['ctrl_jac']['rho']}", flush=True)

# ============ 实验 2：检索 ============
NB = 20 if FAST else 100
NQ = 10 if FAST else 50
rng2 = np.random.default_rng(43)
bank = [rand_pattern(rng2) for _ in range(NB)]
bank_mb = np.stack([mb_encode(p) for p in bank])
bank_ct = np.stack([ctrl_encode(p) for p in bank])
qsel = rng2.choice(NB, NQ, replace=False)

ret = {}
raw_sims = {}
raw_ranks = {}
for noise in [0.1, 0.3]:
    keys = ['mb_cos', 'mb_jac', 'ct_cos', 'ct_jac']
    ranks = {k: [] for k in keys}
    sims_all = {k: [] for k in keys}
    t0 = time.time()
    for bidx in qsel:
        qp = scrambler(bank[bidx], noise, rng2)
        qm = mb_encode(qp)
        qc = ctrl_encode(qp)
        sets = {
            'mb_cos': np.array([cos(qm, b) for b in bank_mb], dtype=np.float32),
            'mb_jac': np.array([jac(qm, b) for b in bank_mb], dtype=np.float32),
            'ct_cos': np.array([cos(qc, b) for b in bank_ct], dtype=np.float32),
            'ct_jac': np.array([jac(qc, b) for b in bank_ct], dtype=np.float32),
        }
        for kk, sims in sets.items():
            order = np.argsort(-sims)
            pos = int(np.where(order == bidx)[0][0])
            ranks[kk].append(pos)
            sims_all[kk].append(sims)
    tab = {}
    for kk in keys:
        r = np.array(ranks[kk])
        tab[kk] = {'top1': round(float((r == 0).mean()), 4), 'top5': round(float((r < 5).mean()), 4)}
    ret[f'noise_{noise}'] = tab
    raw_sims[f'noise_{noise}'] = {kk: np.stack(v) for kk, v in sims_all.items()}
    raw_ranks[f'noise_{noise}'] = {kk: np.array(ranks[kk]) for kk in keys}
    print(f"[exp2] 噪声 {noise}: MB cos top1 {(np.array(ranks['mb_cos'])==0).mean()*100:.1f}% top5 {(np.array(ranks['mb_cos'])<5).mean()*100:.1f}% | MB jac top1 {(np.array(ranks['mb_jac'])==0).mean()*100:.1f}% | CTRL cos top1 {(np.array(ranks['ct_cos'])==0).mean()*100:.1f}% ({time.time()-t0:.1f}s)", flush=True)

# ============ 输出 ============
results = {
    'phase': 4, 'config_used': cfg['config'], 'protocol': {'alpha': ALPHA, 'stim_steps': TSTIM, 'amp': AMP},
    'kc_count': int(NK), 'input_pattern_size': 400,
    'exp1_distance_preservation': {
        'levels': LEVELS, 'n_pairs_per_level': NP_P,
        'input_sim_definition': 'overlap fraction (1 - diff)',
        'per_level_means': {
            'mb_cos': [round(float(v), 4) for v in m_cos.mean(axis=1)],
            'mb_jac': [round(float(v), 4) for v in m_jac.mean(axis=1)],
            'ctrl_cos': [round(float(v), 4) for v in c_cos.mean(axis=1)],
            'ctrl_jac': [round(float(v), 4) for v in c_jac.mean(axis=1)],
        },
        'spearman': sp,
        'criteria_pass': {'mb_cos': sp['mb_cos']['rho'] > 0.7, 'mb_jac': sp['mb_jac']['rho'] > 0.7,
                          'ctrl_cos': sp['ctrl_cos']['rho'] > 0.7, 'ctrl_jac': sp['ctrl_jac']['rho'] > 0.7},
    },
    'exp2_retrieval': ret,
    'exp2_criteria_pass': {
        'mb_cos_noise01_top1>0.8': ret['noise_0.1']['mb_cos']['top1'] > 0.8,
        'mb_jac_noise01_top1>0.8': ret['noise_0.1']['mb_jac']['top1'] > 0.8,
    },
}
results['comparison'] = {
    'exp1_spearman_mb_minus_ctrl_rho': {
        'cos': round(sp['mb_cos']['rho'] - sp['ctrl_cos']['rho'], 4),
        'jac': round(sp['mb_jac']['rho'] - sp['ctrl_jac']['rho'], 4),
    },
    'exp2_top1_mb_minus_ctrl': {
        'noise_0.1_cos': round(ret['noise_0.1']['mb_cos']['top1'] - ret['noise_0.1']['ct_cos']['top1'], 4),
        'noise_0.1_jac': round(ret['noise_0.1']['mb_jac']['top1'] - ret['noise_0.1']['ct_jac']['top1'], 4),
        'noise_0.3_cos': round(ret['noise_0.3']['mb_cos']['top1'] - ret['noise_0.3']['ct_cos']['top1'], 4),
        'noise_0.3_jac': round(ret['noise_0.3']['mb_jac']['top1'] - ret['noise_0.3']['ct_jac']['top1'], 4),
    },
}
json.dump(results, open(f'{OUT}/phase4_results.json', 'w'), indent=2, ensure_ascii=False, default=str)

np.savez_compressed(
    f'{OUT}/phase4_raw.npz',
    levels=np.array(LEVELS, dtype=np.float32),
    input_sim=inp_sim, mb_cos=m_cos, mb_jac=m_jac, ctrl_cos=c_cos, ctrl_jac=c_jac,
    qsel=qsel.astype(np.int32),
    **{f'sims_{n}_{kk}': v for n, d in raw_sims.items() for kk, v in d.items()},
    **{f'ranks_{n}_{kk}': v for n, d in raw_ranks.items() for kk, v in d.items()},
)

with open(f'{OUT}/phase4_pairs.csv', 'w') as f:
    f.write('level,input_sim,mb_cos,mb_jac,ctrl_cos,ctrl_jac\n')
    for li, R in enumerate(LEVELS):
        for k in range(NP_P):
            f.write(f'{R},{inp_sim[li,k]:.3f},{m_cos[li,k]:.4f},{m_jac[li,k]:.4f},{c_cos[li,k]:.4f},{c_jac[li,k]:.4f}\n')

print('PHASE4_SUMMARY ' + json.dumps({
    'exp1_spearman': sp,
    'exp2': ret,
    'criteria': results['exp2_criteria_pass'],
}, ensure_ascii=False), flush=True)