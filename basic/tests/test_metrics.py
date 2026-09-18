#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_metrics.py — A2 指标快项（100 记忆，对照 README）· CI/本地通用版

从脚本位置推导包路径（basic/tests/ → basic/），无需硬编码。
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)          # basic/（脚本位于 basic/tests/ 下）
sys.path.insert(0, PKG)
os.chdir(PKG)                        # 包内数据路径为相对路径（问题#1）：需在包目录运行
from fly_agent import FlyAgent  # noqa: E402


def main():
    t0 = time.time()
    print('== A2 metrics (100 memories) ==')
    fly = FlyAgent(mode='segmented')
    rng = np.random.default_rng(20260918)
    pats = [fly.ctx.rand_pattern(rng) for _ in range(100)]
    keys = [f'm{i}' for i in range(100)]
    for k, p in zip(keys, pats):
        fly.remember(k, p)

    hit = sum(1 for k, p in zip(keys, pats) if (fly.recall(p) or {}).get('key') == k)
    print(f'[A2-1] clean: {hit}/100 = {hit/100:.4f} (README 1.0000) ({time.time()-t0:.0f}s)')

    rngq = np.random.default_rng(555)
    noisy = [fly.ctx.scrambler(p, 0.3, rngq) for p in pats]
    hit = sum(1 for k, q in zip(keys, noisy) if (fly.recall(q) or {}).get('key') == k)
    print(f'[A2-2] noisy0.3: {hit}/100 = {hit/100:.4f} (README 0.99) ({time.time()-t0:.0f}s)')

    for k in keys[:50]:
        fly.disconnect(k)
    bad = sum(1 for k, p in zip(keys[:50], pats[:50]) if (fly.recall(p) or {}).get('key') == k)
    ok = sum(1 for k, p in zip(keys[50:], pats[50:]) if (fly.recall(p) or {}).get('key') == k)
    for k in keys[:50]:
        fly.reconnect(k)
    print(f'[A2-3] disconnect50: bad={bad}/50 ok={ok}/50 (README 0/50,50/50) ({time.time()-t0:.0f}s)')

    fly2 = FlyAgent(mode='segmented')
    pats2 = [fly2.ctx.rand_pattern(rng) for _ in range(25)]
    for i, p in enumerate(pats2):
        fly2.remember(f'mp{i}', p, parts=4)
    hit = 0
    for i, p in enumerate(pats2):
        blocks = [p[j * 100:(j + 1) * 100] for j in range(1)]
        r = fly2.recall_blocked(blocks, k=1)
        if r and r[0]['key'] == f'mp{i}':
            hit += 1
    print(f'[A2-4] decompose25%: {hit}/25 (README 25/25) ({time.time()-t0:.0f}s)')

    hit = 0
    for i, p in enumerate(pats):
        r = fly.recall(p[:200])
        if r and r['key'] == keys[i]:
            hit += 1
    print(f'[A2-5] associate50%: {hit}/100 = {hit/100:.2f} (README 0.91) ({time.time()-t0:.0f}s)')

    rngv = np.random.default_rng(606)
    ins = sum(1 for p in pats if not fly.novelty(fly.ctx.scrambler(p, 0.1, rngv))[0])
    outs = sum(1 for q in [fly.ctx.rand_pattern(rngv) for _ in range(100)] if fly.novelty(q)[0])
    print(f'[A2-6] novelty: in={ins}/100 out={outs}/100 (README 1.00/1.00) ({time.time()-t0:.0f}s)')
    print(f'total {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()