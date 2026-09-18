#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_boundary.py — A3 边界测试 · CI/本地通用版（从脚本位置推导包路径）"""
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
    print('== A3 boundary tests ==')
    fly = FlyAgent(mode='segmented')
    rng = np.random.default_rng(9)
    pats = [fly.ctx.rand_pattern(rng) for _ in range(100)]

    for i, p in enumerate(pats):
        fly.remember(f'b{i}', p)
    try:
        fly.remember('b100', fly.ctx.rand_pattern(rng))
        print('[A3-1] FAIL: no error for 101st memory')
    except RuntimeError as e:
        print(f'[A3-1] OK RuntimeError: {e}')

    pth = os.path.join(HERE, '_tmp_save.npz')
    fly.save(pth)
    fly2 = FlyAgent(mode='segmented')
    fly2.load(pth)
    r1 = (fly.recall(pats[7]) or {}).get('key')
    r2 = (fly2.recall(pats[7]) or {}).get('key')
    print(f'[A3-2] save/load: before={r1} after={r2} -> {"OK" if r1 == r2 == "b7" else "FAIL"}')

    try:
        fly.disconnect('not-exist')
        print('[A3-3] FAIL: no KeyError')
    except KeyError as e:
        print(f'[A3-3] OK KeyError({e})')

    th = getattr(fly, 'novelty_thresh', '?')
    print(f'[A3-4] novelty_thresh = {th}')
    rngv = np.random.default_rng(606)
    ins = [fly.novelty(fly.ctx.scrambler(p, 0.1, rngv))[0] for p in pats[:20]]
    outs = [fly.novelty(fly.ctx.rand_pattern(rngv))[0] for _ in range(20)]
    ok_in = sum(1 for x in ins if x is False) / 20
    ok_out = sum(1 for x in outs if x is True) / 20
    print(f'[A3-4] in-False={ok_in:.2f} out-True={ok_out:.2f} -> {"OK" if ok_in >= 0.9 and ok_out >= 0.9 else "CHECK"}')

    fly3 = FlyAgent(mode='segmented')
    r = fly3.recall(np.arange(400) % 38192)
    print(f'[A3-5a] empty recall -> {r}')
    s = fly3.sleep(rounds=2)
    print(f'[A3-5b] empty sleep -> {s}')
    n = fly3.novelty(np.arange(400) % 38192)
    print(f'[A3-5c] empty novelty -> {n}')
    print(f'total {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()