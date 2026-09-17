#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_ext.py —— 两步判定 + Bootstrap CI 扩展（NREM 判据工具）

对应拍板判据：
- 快筛（单种子集）：rho_s 下降 ≤0.10、t1s 下降 ≤10pp、稀疏率保持 5–15%；稳定性均值上移 >0.05（"苗头"）
- 确认（top 候选，平行种子集）："不显著下降" = 95% CI 重叠；"显著上移" = 区间分离

用法（Python）：
    from eval_ext import stratified_bootstrap_rho, binomial_ci, judge_fast, judge_confirm
用法（CLI）：
    python3 eval_ext.py --demo   # 用 phase5 原始数据演示
"""
import json
import numpy as np
from scipy.stats import spearmanr


def stratified_bootstrap_rho(x, cos, n_levels=5, n_boot=1000, seed=0):
    """按 level 分层重采样的 rho 95% CI。
    x = input_sim（1-R），cos = 编码相似度；数据按 level 连续分块（每块 N/n_levels 个）。"""
    rng = np.random.default_rng(seed)
    n = len(x)
    n_per = n // n_levels
    idx_pool = [np.arange(l * n_per, (l + 1) * n_per) for l in range(n_levels)]
    rho = float(spearmanr(x, cos)[0])
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([pool[rng.integers(0, n_per, n_per)] for pool in idx_pool])
        boot[b] = spearmanr(x[idx], cos[idx])[0]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'rho': round(rho, 4), 'ci95': [round(float(lo), 4), round(float(hi), 4)], 'width': round(float(hi - lo), 4)}


def binomial_ci(p, n, z=1.96):
    """Wilson 区间（比例 CI）。"""
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def judge_fast(base, new, delta_rho=0.10, delta_t1s=0.10, sparse_window=(0.05, 0.15), lift_delta=0.05):
    """快筛判定。base/new = eval 结果 dict（含 distance / retrieval / stability）。"""
    d_rho = new['distance']['rho_s'] - base['distance']['rho_s']
    d_t1s = new['retrieval']['noise_0.1']['t1s'] - base['retrieval']['noise_0.1']['t1s']
    sparse_ok = sparse_window[0] <= new['distance']['sparse_abs'] <= sparse_window[1]
    d_stab = new['stability']['noise_0.05']['cos'] - base['stability']['noise_0.05']['cos']
    return {
        'd_rho': round(d_rho, 4), 'd_t1s': round(d_t1s, 4), 'd_stab': round(d_stab, 4),
        'sparse_ok': bool(sparse_ok),
        'no_regression': bool(d_rho >= -delta_rho and d_t1s >= -delta_t1s and sparse_ok),
        'lift_candidate': bool(d_stab > lift_delta),
    }


def ci_overlap(ci_a, ci_b):
    return min(ci_a[1], ci_b[1]) > max(ci_a[0], ci_b[0])


def judge_confirm(base_ci, new_ci, higher_is_better=True):
    """确认判定：CI 重叠 = not_significant；分离 = significant_up/down。"""
    overlap = ci_overlap(base_ci, new_ci)
    if not overlap:
        if new_ci[1] < base_ci[0]:
            verdict = 'significant_down'
        elif new_ci[0] > base_ci[1]:
            verdict = 'significant_up'
        else:
            verdict = 'not_significant' if higher_is_better else 'significant_down'
    else:
        verdict = 'not_significant'
    return {'overlap': bool(overlap), 'verdict': verdict}


def demo():
    print('== eval_ext demo（phase5 原始数据） ==')
    key = '/sdcard/Download/fly-agent/results/phase5_20260917_ci/phase5_raw.npz'
    npz = np.load(key)
    x = npz['input_sim']
    for tag in ['hyb_t5_i100000', 'kwta_static', 'rp_control']:
        k = f'pairs_cos_{tag}'
        if k not in npz.keys():
            continue
        r = stratified_bootstrap_rho(x, npz[k])
        print(f'{tag}: rho={r["rho"]} 95%CI={r["ci95"]} width={r["width"]}')

    print()
    print('t1s CI 示例（n=50）:', {p: binomial_ci(p, 50) for p in [0.92, 0.54, 1.0]})

    print()
    base = {'distance': {'rho_s': 0.6428, 'sparse_abs': 0.1021},
            'retrieval': {'noise_0.1': {'t1s': 0.92}},
            'stability': {'noise_0.05': {'cos': 0.151}}}
    new = {'distance': {'rho_s': 0.6280, 'sparse_abs': 0.1030},
           'retrieval': {'noise_0.1': {'t1s': 0.94}},
           'stability': {'noise_0.05': {'cos': 0.220}}}
    print('judge_fast 演示:', json.dumps(judge_fast(base, new), ensure_ascii=False))
    print('judge_confirm 演示:', json.dumps(judge_confirm((0.63, 0.72), (0.66, 0.75)), ensure_ascii=False))
    print('judge_confirm 演示2:', json.dumps(judge_confirm((0.63, 0.72), (0.30, 0.45)), ensure_ascii=False))


if __name__ == '__main__':
    import sys
    if '--demo' in sys.argv:
        demo()
    else:
        print(__doc__)