#!/usr/bin/env python3
"""第三阶段：蘑菇体稀疏表征（kWTA / 全局抑制 / 横向抑制 参数扫描）v2

目标：让 MB2 网络（含侧副叶）的 KC 响应「稀疏且可区分」。
- 结构冻结，只调整激活状态（不改 W、不删突触）。
- 参照：FlyHash (arXiv 2001.04907) / FlyModel (arXiv 2107.07617) 的稀疏化思路。
- 全局抑制只统计正激活 mean(max(0, r))，不用 mean(r)（正负抵消的坑）。

验证判据（字面版，三条都满足）：
1) 稀疏度：KC 活跃率（|r|>0.1）在 5%~15%
2) 区分度：不同输入两两响应平均 |Δr| > 0.05
3) 稳定性：同一输入重复 3 次，响应相关系数 > 0.9（确定性重复；另行报告带噪重复指标）

v2 修复：kWTA 从「值阈值」改为「argpartition 精确取 k 个」（原实现因值聚集会多留 10 倍）。
"""
import os
import time
import json
import numpy as np
from scipy.sparse import load_npz, csr_matrix

OUT = os.environ.get('FLY_OUT', 'out')
FAST = os.environ.get('MB_FAST', '0') == '1'
ALPHA = 0.3
TSTIM, STEPS = 5, 12
AMP = 0.5
THRESH_ACT = 0.1
LAT_THR = 0.2
N_PAT = 6 if FAST else 24

W = load_npz(f'{OUT}/mb2_W.npz').tocsr()
N = W.shape[0]
din = np.asarray(W.getnnz(axis=0)).ravel()
kc_thr = np.percentile(din, 99)
KC = np.where(din >= kc_thr)[0]
pool = np.setdiff1d(np.arange(N), KC)
print(f'网络 {N:,} | nnz {W.nnz:,} | KC(p99 扇入>={int(kc_thr)}) {len(KC):,} | 输入池 {len(pool):,}', flush=True)

_pat_rng = np.random.default_rng(3)
patterns = [_pat_rng.choice(pool, 400, replace=False) for _ in range(N_PAT)]


def apply_kwta(s, idx, kpct):
    """精确保留 |s| 最大的 k 个（k = ceil(len*kpct/100)）"""
    vals = np.abs(s[idx])
    m = len(vals)
    k = int(np.ceil(m * kpct / 100.0))
    k = max(1, min(k, m))
    if k >= m:
        return
    keep_idx = np.argpartition(vals, m - k)[m - k:]
    tmp = np.zeros(m, dtype=np.float32)
    sub = s[idx]
    tmp[keep_idx] = sub[keep_idx]
    s[idx] = tmp


def run(gain, stim_idx, mod=None, noise=0.0, seed=0, steps=STEPS):
    mod = mod or {}
    inh = mod.get('inh', 0.0)
    lat = mod.get('lat', 0.0)
    kw_k = mod.get('kwta_k')
    kw_scope = mod.get('kwta_scope', 'kc')
    x = np.zeros(N, dtype=np.float32)
    if stim_idx is not None:
        x[stim_idx] = AMP
    s = np.zeros(N, dtype=np.float32)
    rngn = np.random.default_rng(seed)
    for t in range(steps):
        d = gain * (W @ s)
        if stim_idx is not None and t < TSTIM:
            d = d + x
        if inh:
            d = d - inh * np.float32(np.mean(np.maximum(s, 0.0)))
        if lat:
            d = d - lat * np.float32(np.mean(np.maximum(s - LAT_THR, 0.0)))
        if noise:
            d = d + (noise * rngn.standard_normal(N)).astype(np.float32)
        s = (1 - ALPHA) * s + ALPHA * np.tanh(d)
        if kw_k:
            if kw_scope == 'global':
                apply_kwta(s, np.arange(N), kw_k)
            else:
                apply_kwta(s, KC, kw_k)
    return s


def stab_metrics(gain, mod, pattern, noise, steps=STEPS):
    reps = [run(gain, pattern, mod, noise=noise, seed=11 + 17 * k, steps=steps)[KC] for k in range(3)]
    cs = []
    jac = []
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = reps[i], reps[j]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 0 and nb > 0:
                cs.append(float(np.dot(a, b) / (na * nb)))
            sa = np.abs(a) > THRESH_ACT
            sb = np.abs(b) > THRESH_ACT
            u = (sa | sb).sum()
            jac.append(float((sa & sb).sum() / u) if u else 0.0)
    return (float(np.mean(cs)) if cs else 0.0), (float(np.mean(jac)) if jac else 0.0)


def eval_config(gain, mod, label, steps=STEPS):
    t0 = time.time()
    fulls = []
    resps = []
    for i, p in enumerate(patterns):
        s = run(gain, p, mod, seed=100 + i, steps=steps)
        fulls.append(s)
        resps.append(s[KC])
    resps = np.stack(resps)
    all_act = float(np.mean(np.abs(fulls[0]) > THRESH_ACT))
    act = float(np.mean(np.abs(resps) > THRESH_ACT))
    act05 = float(np.mean(np.abs(resps) > 0.05))
    diffs = []
    cors = []
    for i in range(len(resps)):
        for j in range(i + 1, len(resps)):
            diffs.append(float(np.mean(np.abs(resps[i] - resps[j]))))
            a, b = resps[i], resps[j]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 0 and nb > 0:
                cors.append(float(np.dot(a, b) / (na * nb)))
    disc = float(np.mean(diffs)) if diffs else 0.0
    cosd = float(np.mean(cors)) if cors else 0.0
    sp = [0, min(1, len(patterns) - 1), min(2, len(patterns) - 1)]
    det_cos, det_jac = np.mean([stab_metrics(gain, mod, patterns[i], 0.0, steps) for i in sp], axis=0)
    n05_cos, n05_jac = np.mean([stab_metrics(gain, mod, patterns[i], 0.05, steps) for i in sp], axis=0)
    n02_cos, n02_jac = np.mean([stab_metrics(gain, mod, patterns[i], 0.02, steps) for i in sp], axis=0)
    crit = {'sparse': bool(0.05 <= act <= 0.15), 'discriminate': bool(disc > 0.05), 'stable_det': bool(det_cos > 0.9)}
    met = all(crit.values())
    strict = met and n05_cos > 0.9
    result = {
        'label': label, 'gain': gain, 'mod': mod, 'steps': steps,
        'kc_active': round(act, 4), 'kc_active_005': round(act05, 4), 'all_active': round(all_act, 4),
        'discrimination': round(disc, 5), 'mean_cosine': round(cosd, 4),
        'stab_det_cos': round(float(det_cos), 4), 'stab_det_jaccard': round(float(det_jac), 4),
        'stab_n05_cos': round(float(n05_cos), 4), 'stab_n05_jaccard': round(float(n05_jac), 4),
        'stab_n02_cos': round(float(n02_cos), 4), 'stab_n02_jaccard': round(float(n02_jac), 4),
        'criteria': crit, 'all_met': met, 'strict_noisy_met': bool(strict),
        'eval_s': round(time.time() - t0, 2),
    }
    flag = '★全过' if met else ''
    extra = '★带噪也过' if strict else ''
    print(f"[{label}] KC活跃 {act*100:.1f}% | 全网络 {all_act*100:.1f}% | 区分 {disc:.4f} | 稳定det {det_cos:.3f} | 带噪0.05: cos {n05_cos:.3f}/jac {n05_jac:.3f} {flag}{extra}", flush=True)
    return result


results = []

if FAST:
    base_gains = [1.0, 2.0]
else:
    base_gains = [0.5, 1.0, 2.0, 3.0]

# ---- 0) 基线（不改造）----
baseline_meta = [eval_config(g, {}, f'baseline_g{g}') for g in base_gains]
primary = None
best_sat = -1.0
for r in baseline_meta:
    if r['gain'] in (1.0, 2.0, 3.0) and r['kc_active'] > best_sat:
        best_sat = r['kc_active']
        primary = r['gain']
print(f'>>> primary_gain = {primary} (基线KC饱和率最高 {best_sat*100:.1f}%)', flush=True)
results.extend(baseline_meta)

# ---- 1) kWTA ----
ks = [5, 10] if FAST else [1, 3, 5, 10, 20]
for scope in ['kc', 'global']:
    for k in ks:
        results.append(eval_config(primary, {'kwta_scope': scope, 'kwta_k': k}, f'kwta_{scope}_k{k}'))
if not FAST and primary != 2.0:
    for k in [5, 10]:
        results.append(eval_config(2.0, {'kwta_scope': 'kc', 'kwta_k': k}, f'hedge_kwta_kc_k{k}_g2'))

# ---- 2) 全局抑制（正激活均值）----
inhs = [1, 2] if FAST else [0.5, 1, 2, 5]
for v in inhs:
    results.append(eval_config(primary, {'inh': v}, f'inh_{v}'))
if not FAST:
    for v in [10, 20]:
        results.append(eval_config(primary, {'inh': v}, f'inh_{v}_ext'))
    if primary != 2.0:
        for v in [1, 2, 5]:
            results.append(eval_config(2.0, {'inh': v}, f'hedge_inh_{v}_g2'))

# ---- 3) 横向抑制（仅由强活跃神经元提供的抑制）----
lats = [1] if FAST else [0.5, 1, 2, 5]
for v in lats:
    results.append(eval_config(primary, {'lat': v}, f'lat_{v}'))

# ---- 4) 组合：最佳 kWTA(kc) + 最佳 inh ----


def first_met(prefixes):
    for r in results:
        if any(r['label'].startswith(p) for p in prefixes) and r['all_met']:
            return r
    return None


kw_best = first_met(['kwta_kc_k'])
inh_best = first_met(['inh_'])
if kw_best and inh_best:
    k_best = kw_best['mod']['kwta_k']
    v_best = inh_best['mod']['inh']
    results.append(eval_config(primary, {'kwta_scope': 'kc', 'kwta_k': k_best, 'inh': v_best},
                               f'combo_kwta{k_best}_inh{v_best}'))

# ---- 5) 稳定性调参：低增益/更长步数能否救带噪稳定性 ----
if not FAST:
    for k in [5, 10]:
        for g2 in [0.5, 1.0, 2.0]:
            results.append(eval_config(g2, {'kwta_scope': 'kc', 'kwta_k': k}, f'tune_kwta{k}_g{g2}_s24', steps=24))


# ---- 选最优（字面判据优先；顺序=用户方法顺序）----
def closeness(r):
    pen = 0.0
    if r['kc_active'] < 0.05:
        pen += (0.05 - r['kc_active']) * 20
    if r['kc_active'] > 0.15:
        pen += (r['kc_active'] - 0.15) * 20
    pen += max(0.0, 0.05 - r['discrimination']) * 100
    pen += max(0.0, 0.9 - r['stab_det_cos']) * 10
    return pen


def choose_best(group_prefixes):
    cands = [r for r in results if any(r['label'].startswith(p) for p in group_prefixes) and r['all_met']]
    if not cands:
        return None
    return max(cands, key=lambda r: r['discrimination'])


final = None
for group in [['kwta_kc_k'], ['kwta_global_k'], ['inh_'], ['lat_'], ['combo_'], ['tune_'], ['hedge_']]:
    final = choose_best(group)
    if final:
        break
closest = min(results, key=closeness) if results else None
chosen = final if final else closest

# ---- 保存最佳配置的表征文件 ----
save_info = None
if chosen:
    resp_m = []
    for i, p in enumerate(patterns):
        s = run(chosen['gain'], p, chosen['mod'], seed=100 + i, steps=chosen.get('steps', STEPS))
        resp_m.append(s[KC])
    resp_m = np.stack(resp_m)
    active = np.abs(resp_m) > THRESH_ACT
    csr = csr_matrix(active.astype(np.float32))
    meta = {
        'network': 'mb2_W.npz', 'neurons': int(N), 'nnz': int(W.nnz), 'kc_count': int(len(KC)),
        'config': {'gain': chosen['gain'], 'mod': chosen['mod'], 'steps': chosen.get('steps', STEPS)},
        'metrics': {k: chosen[k] for k in ['kc_active', 'all_active', 'discrimination', 'mean_cosine',
                                           'stab_det_cos', 'stab_n05_cos', 'stab_n05_jaccard']},
        'criteria': chosen['criteria'], 'all_met': chosen['all_met'], 'strict_noisy_met': chosen['strict_noisy_met'],
        'protocol': {'alpha': ALPHA, 'stim_steps': TSTIM, 'steps': chosen.get('steps', STEPS), 'amp': AMP,
                     'active_threshold': THRESH_ACT},
        'note': 'resp = KC 响应（N_PAT 输入 × KC）；active CSR = |resp|>0.1 的稀疏二值支持集；inputs = 每行输入神经元索引',
    }
    np.savez_compressed(f'{OUT}/mb2_sparse_best.npz',
                        kc_idx=KC, inputs=np.stack(patterns), resp=resp_m,
                        csr_data=csr.data, csr_indices=csr.indices, csr_indptr=csr.indptr,
                        csr_shape=np.array(csr.shape))
    json.dump(meta, open(f'{OUT}/mb2_sparse_best_meta.json', 'w'), indent=2, ensure_ascii=False, default=str)
    save_info = meta
    print(f">>> 保存表征文件: {OUT}/mb2_sparse_best.npz ({chosen['label']})", flush=True)

summary = {
    'kc_count': int(len(KC)), 'kc_fanin_threshold': int(kc_thr),
    'primary_gain': primary, 'baseline_best_saturation': round(best_sat, 4),
    'n_configs': len(results), 'final_best': chosen, 'found_all_met': bool(final),
    'found_strict_noisy': any(r.get('strict_noisy_met') for r in results),
    'best_saved_as': save_info is not None,
    'configs': results,
}
json.dump(summary, open(f'{OUT}/mb_sparse_sweep.json', 'w'), indent=2, ensure_ascii=False, default=str)
print('SWEEP_SUMMARY ' + json.dumps({k: v for k, v in summary.items() if k != 'configs'}, ensure_ascii=False), flush=True)