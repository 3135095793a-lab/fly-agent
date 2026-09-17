#!/usr/bin/env python3
"""第五阶段 v2：APL 负反馈回路（补上去相关）

三段扫描 + 双读数 + 三方对比（同种子、同模式，重跑 phase4 全套）：
- spec   ：按任务给定量纲 tau∈[5,20,50] × inh∈[1,3,10]
- scaled ：单位修正后 tau∈[5,20,50] × inh∈[3e3,1e4,3e4]（本模型 drive 量级 O(1e3)）
- hybrid ：APL + kWTA（APL 去相关 + kWTA 取稀疏）
读数：signed（带符号，phase3/4 口径）与 rectified（正发放率，生物学口径）。
判据（与 phase4 同口径，signed）：|s| 活跃率 5-15% 且 ρ>0.7 且 噪声0.1 top-1 >0.85。
"""
import os
import time
import json
import numpy as np
from scipy.sparse import load_npz, csr_matrix
from scipy.stats import spearmanr

OUT = os.environ.get('FLY_OUT', 'out')
FAST = os.environ.get('P5_FAST', '0') == '1'

cfg = json.load(open('ci/configs/mb2_sparse_best_meta.json'))
GAIN = float(cfg['config']['gain'])
KW = int(cfg['config']['mod']['kwta_k'])
STEPS = int(cfg['config']['steps'])
ALPHA = float(cfg['protocol']['alpha'])
TSTIM = int(cfg['protocol']['stim_steps'])
AMP = float(cfg['protocol']['amp'])
TH = 0.1

W = load_npz(f'{OUT}/mb2_W.npz').tocsr()
N = W.shape[0]
din = np.asarray(W.getnnz(axis=0)).ravel()
KC = np.where(din >= np.percentile(din, 99))[0]
pool = np.setdiff1d(np.arange(N), KC)
NK = len(KC)
print(f'网络 {N:,} | KC {NK} | 基准 gain={GAIN} steps={STEPS}', flush=True)

RP = (np.random.default_rng(7).standard_normal((NK, N), dtype=np.float32)) / np.sqrt(400.0)


def kwta(v):
    m = len(v)
    k = max(1, int(np.ceil(m * KW / 100.0)))
    idx = np.argpartition(np.abs(v), m - k)[m - k:]
    o = np.zeros(m, np.float32)
    o[idx] = v[idx]
    return o


def sim_base(x):
    s = np.zeros(N, np.float32)
    for t in range(STEPS):
        s = (1 - ALPHA) * s + ALPHA * np.tanh(GAIN * (W @ s) + (x if t < TSTIM else 0))
    return s


def sim_apl(x, tau, inh):
    s = np.zeros(N, np.float32)
    apl = 0.0
    for t in range(STEPS):
        apl = apl * (1 - 1 / tau) + (1 / tau) * float(np.mean(np.maximum(s[KC], 0.0)))
        d = GAIN * (W @ s) + (x if t < TSTIM else 0)
        d[KC] -= inh * apl
        s = (1 - ALPHA) * s + ALPHA * np.tanh(d)
    return s


def enc_plain(stim):
    x = np.zeros(N, np.float32)
    x[stim] = AMP
    return sim_base(x)[KC].copy()


def enc_kwta(stim):
    x = np.zeros(N, np.float32)
    x[stim] = AMP
    return kwta(sim_base(x)[KC])


def enc_rp(stim):
    x = np.zeros(N, np.float32)
    x[stim] = AMP
    return kwta(RP @ x)


def mk_apl(tau, inh):
    def f(stim):
        x = np.zeros(N, np.float32)
        x[stim] = AMP
        return sim_apl(x, tau, inh)[KC].copy()
    return f


def mk_hybrid(tau, inh):
    def f(stim):
        x = np.zeros(N, np.float32)
        x[stim] = AMP
        return kwta(sim_apl(x, tau, inh)[KC])
    return f


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na * nb > 0 else 0.0


def jac(a, b):
    sa = a != 0
    sb = b != 0
    u = (sa | sb).sum()
    return float((sa & sb).sum() / u) if u else 0.0


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


# ---- 固定模式集（与 phase4 同种子同顺序）----
NP_P = 8 if FAST else 40
LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
rng = np.random.default_rng(42)
PAIRS = []
for R in LEVELS:
    for k in range(NP_P):
        base = rand_pattern(rng)
        part = scrambler(base, R, rng)
        PAIRS.append((R, base, part))

NB = 20 if FAST else 100
NQ = 10 if FAST else 50
rng2 = np.random.default_rng(43)
BANK = [rand_pattern(rng2) for _ in range(NB)]
QSEL = rng2.choice(NB, NQ, replace=False)
QUERIES = []
for noise in [0.1, 0.3]:
    for bidx in QSEL:
        qp = scrambler(BANK[bidx], noise, rng2)
        QUERIES.append((noise, int(bidx), qp))
print(f'模式集: pairs {len(PAIRS)} | bank {NB} | queries {len(QUERIES)}', flush=True)


def build_arms():
    if FAST:
        specs = [(5, 1)]
        scaled = [(20, 10000)]
        hyb = [(20, 30000)]
    else:
        specs = [(t, i) for t in [5, 20, 50] for i in [1, 3, 10]]
        scaled = [(t, i) for t in [5, 20, 50] for i in [3000, 10000, 30000]]
        hyb = [(t, i) for t in [5, 20, 50] for i in [10000, 30000]] + [(5, 100000), (20, 100000)]
    arms = []
    for t, i in specs:
        arms.append((f'spec_t{t}_i{i}', mk_apl(t, i), 'apl_spec', {'tau': t, 'inh': i}))
    for t, i in scaled:
        arms.append((f'apl_t{t}_i{i}', mk_apl(t, i), 'apl', {'tau': t, 'inh': i}))
    for t, i in hyb:
        arms.append((f'hyb_t{t}_i{i}', mk_hybrid(t, i), 'hybrid', {'tau': t, 'inh': i}))
    arms.append(('kwta_static', enc_kwta, 'kwta', {}))
    arms.append(('rp_control', enc_rp, 'rp', {}))
    arms.append(('plain_ref', enc_plain, 'plain', {}))
    return arms


def eval_arm(fenc, tag, kind, params):
    t0 = time.time()
    cb = [fenc(b) for _, b, _ in PAIRS]
    cp = [fenc(p) for _, _, p in PAIRS]
    xs = 1 - np.array([R for R, _, _ in PAIRS])
    n = len(PAIRS)
    # signed
    rc_s = [cos(cb[i], cp[i]) for i in range(n)]
    rho_s = float(spearmanr(xs, rc_s)[0])
    fl_s = float(np.mean([cos(cb[i], cb[j]) for i in range(10) for j in range(i + 1, 10)]))
    allc = np.stack(cb + cp)
    sparse_abs = float(np.mean(np.abs(allc) > TH))
    posf = float(np.mean(allc > TH))
    # rectified
    cbR = [np.maximum(c, 0) for c in cb]
    cpR = [np.maximum(c, 0) for c in cp]
    rc_r = [cos(cbR[i], cpR[i]) for i in range(n)]
    rho_r = float(spearmanr(xs, rc_r)[0])
    fl_r = float(np.mean([cos(cbR[i], cbR[j]) for i in range(10) for j in range(i + 1, 10)]))
    # retrieval
    bank_s = [fenc(p) for p in BANK]
    bank_r = [np.maximum(b, 0) for b in bank_s]
    tab = {}
    for noise in [0.1, 0.3]:
        rs, rr = [], []
        for nz, bidx, qp in QUERIES:
            if nz != noise:
                continue
            qs = fenc(qp)
            qr = np.maximum(qs, 0)
            ss = np.array([cos(qs, b) for b in bank_s])
            sr = np.array([cos(qr, b) for b in bank_r])
            rs.append(int(np.where(np.argsort(-ss) == bidx)[0][0]))
            rr.append(int(np.where(np.argsort(-sr) == bidx)[0][0]))
        rs = np.array(rs)
        rr = np.array(rr)
        tab[f'noise_{noise}'] = {
            't1s': round(float((rs == 0).mean()), 4), 't5s': round(float((rs < 5).mean()), 4),
            't1r': round(float((rr == 0).mean()), 4), 't5r': round(float((rr < 5).mean()), 4),
        }
    crit = {
        'sparse_5_15': bool(0.05 <= sparse_abs <= 0.15),
        'rho_s_gt_0.7': bool(rho_s > 0.7),
        't1s_01_gt_0.85': bool(tab['noise_0.1']['t1s'] > 0.85),
    }
    met = all(crit.values())
    r = {
        'tag': tag, 'kind': kind, 'params': params,
        'sparse_abs': round(sparse_abs, 4), 'posf': round(posf, 4),
        'rho_s': round(rho_s, 4), 'rho_r': round(rho_r, 4),
        'floor_s': round(fl_s, 4), 'floor_r': round(fl_r, 4),
        'exp2': tab, 'criteria': crit, 'all_met': met,
        'eval_s': round(time.time() - t0, 1),
    }
    flag = '★全过' if met else ''
    print(f"[{tag}] |s|{sparse_abs*100:.1f}% pos{posf*100:.1f}% | ρs{rho_s:.3f} fls{fl_s:.2f} | ρr{rho_r:.3f} flr{fl_r:.2f} | t1s@0.1 {tab['noise_0.1']['t1s']*100:.0f}% t1s@0.3 {tab['noise_0.3']['t1s']*100:.0f}% {flag}", flush=True)
    return r


arms = build_arms()
results = [eval_arm(f, t, k, p) for t, f, k, p in arms]

# ---- 选择 ----


def met(r):
    return r['all_met']


met_list = [r for r in results if met(r)]
proposed = [r for r in results if r['kind'] in ('apl', 'hybrid')]
proposed_met = [r for r in proposed if met(r)]


def pen(r):
    p = 0.0
    if r['sparse_abs'] < 0.05:
        p += (0.05 - r['sparse_abs']) * 20
    if r['sparse_abs'] > 0.15:
        p += (r['sparse_abs'] - 0.15) * 20
    p += max(0.0, 0.7 - r['rho_s']) * 100
    p += max(0.0, 0.85 - r['exp2']['noise_0.1']['t1s']) * 100
    return p


if proposed_met:
    best = max(proposed_met, key=lambda r: (r['exp2']['noise_0.1']['t1s'], r['rho_s']))
else:
    best = min(proposed, key=pen)

pure = [r for r in results if r['kind'] == 'apl']
pure_met = [r for r in pure if met(r)]
def pure_pen(r):
    return abs(r['sparse_abs'] - 0.1) + max(0.0, 0.7 - r['rho_s']) + max(0.0, 0.85 - r['exp2']['noise_0.1']['t1s'])


best_pure = max(pure_met, key=lambda r: (r['exp2']['noise_0.1']['t1s'], r['rho_s'])) if pure_met else min(pure, key=pure_pen)
print(f">>> best: {best['tag']} (met={best['all_met']}) | best_pure_apl: {best_pure['tag']} (met={bool(pure_met)})", flush=True)

# ---- 对照表 ----


def slim(r):
    return {
        'tag': r['tag'], 'kind': r['kind'],
        'sparse_abs': r['sparse_abs'], 'posf': r['posf'],
        'rho_s': r['rho_s'], 'rho_r': r['rho_r'], 'floor_s': r['floor_s'],
        't1s_01': r['exp2']['noise_0.1']['t1s'], 't5s_01': r['exp2']['noise_0.1']['t5s'],
        't1s_03': r['exp2']['noise_0.3']['t1s'], 't5s_03': r['exp2']['noise_0.3']['t5s'],
        'all_met': r['all_met'],
    }


def get(tag):
    return next(r for r in results if r['tag'] == tag)


comparison = {
    'best': slim(best), 'best_pure_apl': slim(best_pure),
    'kwta_static': slim(get('kwta_static')), 'rp_control': slim(get('rp_control')),
    'plain_ref': slim(get('plain_ref')),
}
phase4_ref = {'mb_kwta_rho_cos': 0.5715, 'mb_kwta_top1_n01': 0.86, 'rp_rho_cos': 0.952, 'rp_top1_n01': 1.0}

# ---- 输出 ----


def enc_of_tag(tag):
    for t, f, k, p in arms:
        if t == tag:
            return f
    raise KeyError(tag)


def pairs_arrays(fenc):
    cb = [fenc(b) for _, b, _ in PAIRS]
    cp = [fenc(p) for _, _, p in PAIRS]
    ac = np.empty(len(PAIRS), np.float32)
    aj = np.empty(len(PAIRS), np.float32)
    for i in range(len(PAIRS)):
        ac[i] = cos(cb[i], cp[i])
        aj[i] = jac(cb[i], cp[i])
    return ac, aj


raw_tags = [best['tag']]
if best_pure['tag'] != best['tag']:
    raw_tags.append(best_pure['tag'])
raw_tags += ['kwta_static', 'rp_control']
raw = {}
cache = {}
for tag in raw_tags:
    ac, aj = pairs_arrays(enc_of_tag(tag))
    cache[tag] = (ac, aj)
    raw[f'pairs_cos_{tag}'] = ac
    raw[f'pairs_jac_{tag}'] = aj
raw['levels'] = np.array([R for R, _, _ in PAIRS], np.float32)
raw['input_sim'] = 1.0 - raw['levels']
np.savez_compressed(f'{OUT}/phase5_raw.npz', **raw)

with open(f'{OUT}/phase5_pairs.csv', 'w') as f:
    f.write('arm,level,pair_idx,input_sim,cos,jac\n')
    for tag in raw_tags:
        ac, aj = cache[tag]
        for i, (R, _, _) in enumerate(PAIRS):
            f.write(f'{tag},{R},{i},{1-R:.3f},{ac[i]:.4f},{aj[i]:.4f}\n')

# 表征文件（best 臂, bank 模式）
fenc_b = enc_of_tag(best['tag'])
bankc = np.stack([fenc_b(p) for p in BANK])
csr = csr_matrix((np.abs(bankc) > TH).astype(np.float32))
np.savez_compressed(f'{OUT}/mb2_sparse_best_apl.npz',
                    kc_idx=KC, inputs=np.stack(BANK), resp=bankc,
                    csr_data=csr.data, csr_indices=csr.indices, csr_indptr=csr.indptr,
                    csr_shape=np.array(csr.shape))
json.dump({'config_tag': best['tag'], 'metrics': slim(best), 'protocol': {'alpha': ALPHA, 'stim_steps': TSTIM, 'steps': STEPS, 'amp': AMP, 'gain': GAIN, 'kwta_k': KW}},
          open(f'{OUT}/mb2_sparse_best_apl_meta.json', 'w'), indent=2, ensure_ascii=False, default=str)

summary = {
    'phase': 5, 'fast': FAST,
    'best': slim(best), 'best_pure_apl': slim(best_pure),
    'pure_apl_any_met': bool(pure_met),
    'n_met_total': len(met_list),
    'n_met_proposed': len(proposed_met),
    'comparison': comparison, 'phase4_reference': phase4_ref,
    'arms': results,
    'protocol': {'alpha': ALPHA, 'stim_steps': TSTIM, 'steps': STEPS, 'amp': AMP, 'gain': GAIN, 'kwta_k': KW},
}
json.dump(summary, open(f'{OUT}/phase5_results.json', 'w'), indent=2, ensure_ascii=False, default=str)
print('PHASE5_SUMMARY ' + json.dumps({'best': slim(best), 'best_pure_apl': slim(best_pure), 'pure_met': bool(pure_met), 'n_met': len(met_list)}, ensure_ascii=False, default=str), flush=True)