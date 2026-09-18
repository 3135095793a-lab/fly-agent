#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_sleep.py — sleep(20, σ2, struct=False) 后加噪检索（README 声称 0.98~1.00）

从脚本位置推导包路径（basic/tests/ → basic/）。
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
os.chdir(PKG)
from fly_agent import FlyAgent  # noqa: E402


def main():
    t0 = time.time()
    print('== sleep(20, v2, sigma=2) test ==')
    fly = FlyAgent(mode='segmented')
    rng = np.random.default_rng(20260918)
    pats = [fly.ctx.rand_pattern(rng) for _ in range(100)]
    keys = [f'm{i}' for i in range(100)]
    for k, p in zip(keys, pats):
        fly.remember(k, p)
    rngq = np.random.default_rng(555)
    noisy = [fly.ctx.scrambler(p, 0.3, rngq) for p in pats]

    pre = sum(1 for k, q in zip(keys, noisy) if (fly.recall(q) or {}).get('key') == k)
    print(f'[sleep] pre-sleep noisy0.3: {pre}/100')

    sl = fly.sleep(rounds=20, lam=0.90, noise_sigma=2.0, struct_update=False)
    print(f'[sleep] done: {sl}  ({time.time()-t0:.0f}s)')

    hit = sum(1 for k, q in zip(keys, noisy) if (fly.recall(q) or {}).get('key') == k)
    print(f'[sleep] post-sleep noisy0.3: {hit}/100 = {hit/100:.2f} (README 0.98~1.00) ({time.time()-t0:.0f}s)')

    clean = sum(1 for k, p in zip(keys, pats) if (fly.recall(p) or {}).get('key') == k)
    print(f'[sleep] post-sleep clean: {clean}/100')
    print(f'total {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()