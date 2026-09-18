#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""demo.py — 果蝇 agent 基础版 · 全能力演示

跑法（在 fly-agent-basic 目录下）：
    python3 demo.py

约 2 分钟。睡眠演示用 rounds=2（快速）；完整效果用 rounds=20。
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
from fly_agent import FlyAgent  # noqa: E402


def line(t):
    print(chr(10) + '=' * 58)
    print(t)
    print('=' * 58)


def main():
    t00 = time.time()
    line('果蝇 agent 基础版 · 全能力演示')
    fly = FlyAgent(mode='segmented')
    print('模式: segmented | KC 空间: %d | 段: %d x %d' % (
        len(fly.kc_all), fly.n_segments, fly.seg_len))

    rng = np.random.default_rng(42)
    pats = [fly.ctx.rand_pattern(rng) for _ in range(12)]
    keys = ['m%d' % i for i in range(12)]

    # 1. 存与取
    line('[1] 存 12 个记忆 → 干净检索')
    for k, p in zip(keys, pats):
        fly.remember(k, p)
    hit = sum(1 for k, p in zip(keys, pats) if (fly.recall(p) or {}).get('key') == k)
    print('干净检索: %d/12' % hit)

    # 2. 加噪检索
    line('[2] 加噪 0.3 检索（噪声鲁棒性）')
    rngq = np.random.default_rng(7)
    noisy = [fly.ctx.scrambler(p, 0.3, rngq) for p in pats]
    hit = sum(1 for k, q in zip(keys, noisy) if (fly.recall(q) or {}).get('key') == k)
    print('加噪检索: %d/12' % hit)

    # 3. 联想（部分线索）
    line('[3] 联想：50% 输入线索 → 召回完整记忆')
    rngq2 = np.random.default_rng(11)
    partial = []
    for p in pats:
        half = rngq2.choice(p, len(p) // 2, replace=False)
        partial.append(np.sort(half))
    hit = 0
    for k, q in zip(keys, partial):
        r = fly.recall(q)
        hit += int(r and r['key'] == k)
    print('50%% 线索召回: %d/12' % hit)

    # 4. 断连与接回
    line('[4] 断连与接回（隔离门控）')
    fly.disconnect('m3', mode='hard')
    r = fly.recall(pats[3])
    print('断开 m3 后检索 m3 的输入 -> %s（应为 None 或其它 key）' % (r['key'] if r else None))
    r2 = fly.recall(pats[4])
    print('同时检索 m4 -> %s（不受影响）' % (r2['key'] if r2 else None))
    fly.reconnect('m3')
    r = fly.recall(pats[3])
    print('接回后 -> %s' % (r['key'] if r else None))

    # 5. 分辨异样
    line('[5] 分辨异样（novelty）')
    fresh = fly.ctx.scrambler(pats[0], 0.1, np.random.default_rng(3))
    is_novel, sim = fly.novelty(fresh)
    print('库内记忆的变体: is_novel=%s, max_sim=%.3f' % (is_novel, sim))
    alien = fly.ctx.rand_pattern(np.random.default_rng(999))
    is_novel, sim = fly.novelty(alien)
    print('全新随机模式:   is_novel=%s, max_sim=%.3f' % (is_novel, sim))

    # 6. 创新组合
    line('[6] 创新组合（recombine）')
    fly.recombine('combo', 'm0', 'm1', split=0.5)
    r = fly.recall(pats[0])
    print('组合后检索 m0 输入 -> %s' % (r['key'] if r else None))
    print('记忆总数: %d' % len(fly.memory))

    # 7. 睡眠（快速版）
    line('[7] 睡眠 v2（rounds=2 快速演示；完整用 rounds=20）')
    t0 = time.time()
    sl = fly.sleep(rounds=2, lam=0.90, noise_sigma=2.0, struct_update=False)
    print('sleep 完成: g_mean=%.4f  (%.0fs)' % (sl['g_mean'], time.time() - t0))
    hit = sum(1 for k, q in zip(keys, noisy) if (fly.recall(q) or {}).get('key') == k)
    print('睡后加噪检索: %d/12' % hit)

    # 8. 总结
    line('演示完成')
    st = fly.stats()
    print('stats: %s' % st)
    print('总耗时: %.0fs' % (time.time() - t00))


if __name__ == '__main__':
    main()

