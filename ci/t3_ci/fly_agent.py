#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fly_agent.py — 果蝇 Agent（v1.1：双模式 · global + segmented 隔离版）

模式：
- mode='global'（v1 行为）：全局编码（hybrid+kwta），输入列 mask 门控
- mode='segmented'（阶段 7.2「隔离」）：
    KC 空间分段（每段 seg_len 个神经元），每个记忆的码物理限制在自己段内；
    disconnect = 段级门控（只关该段的读数通道）→ 零误伤（网络状态不动）。

设计：AGENT-DESIGN.md + 阶段 7.2 任务书。接口（两模式一致）：
    remember / recall / recall_topk / sleep / disconnect / reconnect /
    list_disconnected / stats / save / load
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'dream'))

from eval_kit import EvalContext, positivize  # noqa: E402
import nrem_v1 as NREM  # noqa: E402


class FlyAgent:
    def __init__(self, net_path=None, mode='global', segments=100, seg_k=None, seg_len=None,
                 kc_pct=None, quiet=False):
        self.quiet = quiet
        self.mode = mode
        self.segments = segments
        self.seg_k = seg_k
        self.requested_seg_len = seg_len
        self.kc_pct = kc_pct
        self.ctx = EvalContext(net_path=net_path)  # net 路径：显式参数 > FLY_NET env > protocol 默认
        coo = self.ctx.W.tocoo()
        self._pre = coo.row
        self._post = coo.col
        self.ctx._pre = coo.row
        self.ctx._post = coo.col
        self.n_edges = self.ctx.W.nnz

        self.g = np.ones(self.n_edges, np.float32)
        self.memory = {}
        self.ban_count = {}
        self.sleep_rounds_total = 0
        self.sleep_history = []
        self._Wq_cache = {}

        self.tau = 5.0
        self.inh = 100000.0

        if self.mode == 'segmented':
            if self.kc_pct is None:
                kc_all = np.arange(self.ctx.N)          # 全部输出神经元（7.2 定案）
            else:
                din = np.asarray(self.ctx.W.getnnz(axis=0)).ravel()
                thr = np.percentile(din, self.kc_pct)
                kc_all = np.where(din >= thr)[0]
            rng = np.random.default_rng(909)
            kc_all = kc_all[rng.permutation(len(kc_all))]
            if self.requested_seg_len:
                L = int(self.requested_seg_len)
                S = len(kc_all) // L
            else:
                S = min(self.segments, len(kc_all))
                L = max(1, len(kc_all) // S)
            self.kc_all = kc_all[: S * L]
            self.n_segments = int(S)
            self.seg_len = int(L)
            self.seg_k = self.seg_k or max(1, int(L * 0.2))
            self.seg_enabled = np.ones(self.n_segments, np.float32)  # 段门控值（1.0=开 / 0.0=hard断 / 0.2=soft断）
        else:
            self.kc_all = self.ctx.KC
            self.n_segments = 0
            self.seg_enabled = np.zeros(0, np.float32)

        self._ban_mask = np.zeros(self.ctx.N, bool)
        self._refresh_eff()

    # ---------------- 内部：模式与编码 ----------------
    def _to_pattern(self, data):
        if isinstance(data, np.ndarray) and data.dtype.kind in 'iu' and data.ndim == 1:
            # 神经元索引数组（任意长度 ≤2000、值域合法）——支持"部分输入"查询
            if data.size and data.size <= 2000 and int(data.max()) < self.ctx.N and int(data.min()) >= 0:
                return data.astype(np.int64)
        v = np.asarray(data, dtype=np.float32).ravel()
        return self._vec_to_pattern(v)

    def _vec_to_pattern(self, v):
        D = int(v.shape[0])
        Wq = self._Wq_cache.get(D)
        if Wq is None:
            rng = np.random.default_rng(101 + D)
            Wq = (rng.standard_normal((400, D), dtype=np.float32) / np.sqrt(D)).astype(np.float32)
            self._Wq_cache[D] = Wq
        q = Wq @ v
        qn = (q - q.min()) / (q.max() - q.min() + 1e-9)
        pool = self.ctx.pool
        part = len(pool) // 400
        starts = np.arange(400) * part
        offs = np.floor(qn * (part - 1)).astype(np.int64)
        return pool[starts + offs]

    def _encode_s(self, pattern):
        x = self.ctx.stim_vec(pattern)
        return self.ctx.sim_apl(x, self.tau, self.inh)

    def _encode(self, pattern):
        """global 码：hybrid 动力学 + KC 读出 kWTA + 极性翻正。"""
        s = self._encode_s(pattern)
        return positivize(self.ctx.kwta(s[self.ctx.KC]))

    def _encode_full(self, pattern):
        """segmented 全段码 (S, L)：段内 |s| top-k（raw 值，约 20% 稀疏）——未应用段门控。"""
        s = self._encode_s(pattern)
        vals = s[self.kc_all].reshape(self.n_segments, self.seg_len)
        L = self.seg_len
        k = min(self.seg_k, L)
        idx = np.argpartition(np.abs(vals), L - k, axis=1)[:, L - k:]
        out = np.zeros_like(vals)
        rows = np.arange(self.n_segments)[:, None]
        out[rows, idx] = vals[rows, idx]
        return out

    def _apply_gate(self, qfull):
        q = qfull.copy()
        q *= self.seg_enabled[:, None]
        return q

    @staticmethod
    def _cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0

    # ---------------- 内部：门控（global 模式用） ----------------
    def _refresh_eff(self):
        mask_edges = ~self._ban_mask[self._post]
        g_eff = np.where(mask_edges, self.g, 0.0).astype(np.float32)
        self.ctx.apply_gain(g_eff)
        self._g_eff = g_eff

    def _rebuild_ban(self):
        m = np.zeros(self.ctx.N, bool)
        if self.ban_count:
            idx = np.array([n for n, c in self.ban_count.items() if c > 0], dtype=np.int64)
            if idx.size:
                m[idx] = True
        self._ban_mask = m

    # ---------------- 公共接口 ----------------
    def remember(self, key, data, parts=1):
        key = str(key)
        pattern = self._to_pattern(data)
        if self.mode == 'segmented':
            if parts > 1:
                # 多段记忆（decompose）：输入分 parts 块，每块编码到一段
                n = len(pattern)
                blk = n // parts
                sub_patterns = [pattern[i * blk:(i + 1) * blk] for i in range(parts)]
                start = len(self.memory) * parts
                if start + parts > self.n_segments:
                    raise RuntimeError(f'segment slots exhausted for parts={parts} (need {start + parts} > {self.n_segments})')
                codes = [self._encode_full(sp)[start + i].copy() for i, sp in enumerate(sub_patterns)]
                self.memory[key] = {'pattern': pattern, 'sub_patterns': sub_patterns, 'codes': np.stack(codes),
                                    'slots': list(range(start, start + parts)), 'parts': parts, 'data': data}
                return {'key': key, 'parts': parts, 'start_slot': start}
            slot = len(self.memory)
            if slot >= self.n_segments:
                raise RuntimeError(f'segment slots exhausted ({self.n_segments} 段 < {slot + 1} 记忆)')
            code = self._encode_full(pattern)[slot].copy()
            self.memory[key] = {'pattern': pattern, 'code': code, 'data': data, 'slot': slot}
            return {'key': key, 'code_dim': int(code.shape[0]), 'slot': slot}
        code = self._encode(pattern)
        self.memory[key] = {'pattern': pattern, 'code': code, 'data': data}
        return {'key': key, 'code_dim': int(code.shape[0])}

    def recall(self, query):
        r = self.recall_topk(query, k=1)
        return r[0] if r else None

    def recall_topk(self, query, k=5):
        keys = list(self.memory.keys())
        if not keys:
            return []
        if self.mode == 'segmented':
            qfull = self._apply_gate(self._encode_full(self._to_pattern(query)))
            sims = np.array([self._cos(qfull[self.memory[kk]['slot']], self.memory[kk]['code']) for kk in keys])
        else:
            q = self._encode(self._to_pattern(query)).astype(np.float64)
            qn = q / (np.linalg.norm(q) + 1e-9)
            codes = np.stack([self.memory[kk]['code'] for kk in keys])
            cn = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-9)
            sims = cn @ qn
        order = np.argsort(-sims)[:k]
        return [{'key': keys[i], 'data': self.memory[keys[i]]['data'], 'score': float(sims[i])} for i in order]

    def recall_blocked(self, blocks, k=1):
        """多段记忆（parts>1）的部分块召回：blocks=已提供的块列表（按块顺序）。"""
        if self.mode != 'segmented':
            return []
        keys = [kk for kk, m in self.memory.items() if m.get('parts', 1) > 1]
        if not keys:
            return []
        scores = []
        for kk in keys:
            m = self.memory[kk]
            scs = []
            for i, blk in enumerate(blocks):
                if i >= m['parts']:
                    break
                if float(self.seg_enabled[m['slots'][i]]) < 1.0:
                    continue
                q = self._encode_full(blk)[m['slots'][i]]
                scs.append(self._cos(q, m['codes'][i]))
            scores.append(float(np.mean(scs)) if scs else 0.0)
        order = np.argsort(-np.array(scores))[:k]
        return [{'key': keys[i], 'data': self.memory[keys[i]]['data'], 'score': float(scores[i])} for i in order]

    def novelty(self, query, thresh=None):
        """异样检测 -> (is_novel, max_sim)。"""
        keys = list(self.memory.keys())
        if not keys:
            return (True, 0.0)
        if thresh is None:
            thresh = getattr(self, 'novelty_thresh', 0.5)
        if self.mode == 'segmented':
            qfull = self._apply_gate(self._encode_full(self._to_pattern(query)))
            sims = [self._cos(qfull[self.memory[kk]['slot']], self.memory[kk]['code'])
                    for kk in keys if 'slot' in self.memory[kk]]
        else:
            q = self._encode(self._to_pattern(query)).astype(np.float64)
            qn = q / (np.linalg.norm(q) + 1e-9)
            codes = np.stack([self.memory[kk]['code'] for kk in keys])
            cn = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-9)
            sims = (cn @ qn).tolist()
        mx = max(sims) if sims else 0.0
        return (bool(mx < thresh), float(mx))

    def recombine(self, new_key, key1, key2, split=0.5):
        """创新组合：前半用 key1 的输入、后半用 key2 的输入 → 存为新记忆。"""
        m1 = self.memory[str(key1)]
        m2 = self.memory[str(key2)]
        p1, p2 = m1['pattern'], m2['pattern']
        cut = int(len(p1) * split)
        new_pat = np.concatenate([p1[:cut], p2[cut:]])
        return self.remember(new_key, new_pat)

    def sleep(self, rounds=20, lam=0.90, rule='A', noise_sigma=0.0, noise_seed=7003, noise_mode='lognormal',
              struct_update=True):
        """做梦巩固。struct_update=False（NREM v2）：跳过共激活统计+Δg，仅每轮 g*=lam + 末尾 σ 扰动。"""
        keys = list(self.memory.keys())
        if not keys:
            return {'skipped': 'empty memory'}
        x_cache = [self.ctx.stim_vec(self.memory[kk]['pattern']) for kk in keys]
        n_exp = len(x_cache)
        s = np.zeros(self.ctx.N, np.float32)
        order_rng = np.random.default_rng(7001)
        noise_rng = np.random.default_rng(7002)
        traj = []
        for r in range(rounds):
            if struct_update:
                order = order_rng.permutation(n_exp)
                s, stat, T, diag = NREM.sleep_round(self.ctx, s, x_cache, order, noise_rng, NREM.CFG['sigma'], rule)
                dg = NREM.compute_delta_g(stat, T, n_exp, rule, NREM.CFG)
                self.g = NREM.apply_update(self.g, dg, lam, NREM.CFG)
            else:
                # NREM v2：仅下缩放（增益重标定），不做结构学习
                self.g = np.clip(self.g * np.float32(lam), 0.0, NREM.CFG['g_max']).astype(np.float32)
            traj.append({'round': r + 1, 'g_mean': float(self.g.mean())})
        self.sleep_rounds_total += rounds
        self.sleep_history.append({'rounds': rounds, 'lam': lam, 'traj': traj})
        # 7.3 T3 修复：睡后对 g 施加受控乘性扰动（恢复增益异质性；noise_sigma=0 时行为不变）
        if noise_sigma and noise_sigma > 0:
            rng = np.random.default_rng(noise_seed)
            z = rng.standard_normal(self.n_edges).astype(np.float32)
            if noise_mode == 'linear':
                self.g = (self.g * (1.0 + noise_sigma * z)).astype(np.float32)
            else:  # lognormal（默认；保证正值）
                self.g = (self.g * np.exp(noise_sigma * z)).astype(np.float32)
            self.g = np.clip(self.g, 0.0, NREM.CFG['g_max']).astype(np.float32)
        self._refresh_eff()
        # 睡后刷新库码（保持库/查询同编码口径）
        if self.mode == 'segmented':
            for kk in self.memory:
                full = self._encode_full(self.memory[kk]['pattern'])
                self.memory[kk]['code'] = full[self.memory[kk]['slot']].copy()
        else:
            for kk in self.memory:
                self.memory[kk]['code'] = self._encode(self.memory[kk]['pattern'])
        return {'rounds': rounds, 'g_mean': float(self.g.mean()),
                'g_p99': float(np.percentile(self.g, 99)), 'n_replayed': n_exp}

    def disconnect(self, key, mode='hard'):
        k = str(key)
        if k not in self.memory:
            raise KeyError(k)
        if self.mode == 'segmented':
            slot = self.memory[k]['slot']
            self.seg_enabled[slot] = 0.0 if mode == 'hard' else 0.2
            return {'disconnected': k, 'mechanism': f'segment-gate-{mode}', 'slot': int(slot)}
        for n in self.memory[k]['pattern']:
            n = int(n)
            self.ban_count[n] = self.ban_count.get(n, 0) + 1
        self._rebuild_ban()
        self._refresh_eff()
        return {'disconnected': k, 'mechanism': 'input-column-mask'}

    def reconnect(self, key):
        k = str(key)
        if k not in self.memory:
            raise KeyError(k)
        if self.mode == 'segmented':
            self.seg_enabled[self.memory[k]['slot']] = 1.0
            return {'reconnected': k, 'mechanism': 'segment-gate'}
        for n in self.memory[k]['pattern']:
            n = int(n)
            self.ban_count[n] = max(0, self.ban_count.get(n, 0) - 1)
        self._rebuild_ban()
        self._refresh_eff()
        return {'reconnected': k, 'mechanism': 'input-column-mask'}

    def list_disconnected(self):
        if self.mode == 'segmented':
            return [k for k, m in self.memory.items() if float(self.seg_enabled[m['slot']]) < 1.0]
        out = []
        for k, m in self.memory.items():
            if all(self.ban_count.get(int(n), 0) > 0 for n in m['pattern']):
                out.append(k)
        return out

    def stats(self):
        keys = list(self.memory.keys())
        if not keys:
            sp = 0.0
        elif self.mode == 'segmented':
            sp = float(np.mean([np.mean(np.abs(m['code']) > 1e-6) for m in self.memory.values()]))
        else:
            codes = np.stack([self.memory[kk]['code'] for kk in keys])
            sp = float((np.abs(codes) > 0.1).mean())
        return {
            'mode': self.mode,
            'n_memories': len(keys),
            'n_segments': self.n_segments,
            'seg_len': self.seg_len if self.mode == 'segmented' else None,
            'seg_k': self.seg_k if self.mode == 'segmented' else None,
            'kc_count': int(len(self.kc_all)),
            'g_mean': round(float(self.g.mean()), 6),
            'g_p99': round(float(np.percentile(self.g, 99)), 6),
            'code_sparsity': round(sp, 4),
            'sleep_rounds_total': self.sleep_rounds_total,
            'n_disconnected': len(self.list_disconnected()),
        }

    def save(self, path):
        keys = list(self.memory.keys())
        state = {
            'mode': np.array([self.mode], dtype=object),
            'kc_pct': np.array([self.kc_pct if self.kc_pct is not None else -1]),
            'seg_len': np.array([self.seg_len if self.mode == 'segmented' else 0]),
            'seg_k': np.array([int(self.seg_k or 0)]),
            'kc_all': self.kc_all,
            'seg_enabled': self.seg_enabled,
            'g': self.g,
            'keys': np.array(keys, dtype=object),
            'patterns': np.stack([self.memory[kk]['pattern'] for kk in keys]) if keys else np.zeros((0, 400), np.int64),
            'codes': np.stack([self.memory[kk]['code'] for kk in keys]) if keys else np.zeros((0, 1), np.float32),
            'datas': np.array([self.memory[kk]['data'] for kk in keys], dtype=object) if keys else np.array([], dtype=object),
            'slots': np.array([self.memory[kk].get('slot', -1) for kk in keys]) if keys else np.zeros(0, np.int64),
            'ban': np.array(list(self.ban_count.items()), dtype=object) if self.ban_count else np.zeros((0, 2), dtype=object),
            'sleep_rounds_total': np.array([self.sleep_rounds_total]),
        }
        np.savez(path, **state)
        return path

    def load(self, path):
        z = np.load(path, allow_pickle=True)
        self.mode = str(z['mode'][0]) if 'mode' in z.files else 'global'
        kv = int(z['kc_pct'][0]) if 'kc_pct' in z.files else 95
        self.kc_pct = kv if kv >= 0 else None
        self.seg_len = int(z['seg_len'][0]) if 'seg_len' in z.files else 16
        sk = int(z['seg_k'][0]) if 'seg_k' in z.files else 0
        self.seg_k = sk if sk > 0 else None
        self.kc_all = z['kc_all'].astype(np.int64) if 'kc_all' in z.files else self.ctx.KC
        self.seg_enabled = z['seg_enabled'].astype(np.float32) if 'seg_enabled' in z.files else np.zeros(0, np.float32)
        if self.mode == 'segmented':
            self.n_segments = int(len(self.kc_all) // self.seg_len)
        self.g = z['g'].astype(np.float32)
        keys = list(z['keys'])
        patterns = z['patterns']
        codes = z['codes']
        datas = list(z['datas'])
        slots = z['slots'] if 'slots' in z.files else np.full(len(keys), -1)
        self.memory = {}
        for i, kk in enumerate(keys):
            entry = {'pattern': patterns[i].astype(np.int64), 'code': codes[i].astype(np.float32), 'data': datas[i]}
            if int(slots[i]) >= 0:
                entry['slot'] = int(slots[i])
            self.memory[str(kk)] = entry
        self.ban_count = {int(n): int(c) for n, c in z['ban']} if z['ban'].size else {}
        self.sleep_rounds_total = int(z['sleep_rounds_total'][0])
        self._rebuild_ban()
        self._refresh_eff()
        return {'loaded': len(keys), 'mode': self.mode}


if __name__ == '__main__':
    # 冒烟（两模式）：存 3 → 查 → 断 → 接
    for mode in ['global', 'segmented']:
        print(f'--- smoke: mode={mode} ---')
        fly = FlyAgent(mode=mode)
        rng = np.random.default_rng(0)
        pats = [fly.ctx.rand_pattern(rng) for _ in range(3)]
        for i, p in enumerate(pats):
            fly.remember(f'k{i}', p)
        r = fly.recall(pats[1])
        print('recall ->', r['key'], round(r['score'], 4))
        fly.disconnect('k1')
        r2 = fly.recall(pats[1])
        print('after disconnect ->', r2['key'], round(r2['score'], 4))
        fly.reconnect('k1')
        r3 = fly.recall(pats[1])
        print('after reconnect ->', r3['key'], round(r3['score'], 4))
        print('stats:', fly.stats())