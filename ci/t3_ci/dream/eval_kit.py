#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dream/eval_kit.py — fly-agent 做梦阶段统一评测套件 v1

用法（Python API）：
    from eval_kit import EvalContext, evaluate_encoder, make_plain, make_kwta_readout, make_hybrid
    ctx = EvalContext()                      # 读 dream/protocol.json + data/mb2_W.npz
    enc = make_hybrid(ctx, tau=5, inh=100000)
    result = evaluate_encoder(ctx, enc, name="hyb_t5_i100000")
    # result 为 dict，可直接 json.dump 落盘

用法（CLI）：
    python3 eval_kit.py --selftest           # 历史校验点回归（全部）
    python3 eval_kit.py --selftest --quick   # 只跑必要项

编码器契约（详见 protocol.json）：
    encoder(stim_indices, noise=0.0, seed=None) -> np.ndarray (KC 维, float32, 发放率语义)
    noise>0：每步驱动 d 上加 noise*standard_normal(N)（与 mb_sparse 稳定性协议一致）

四项评测：distance_preservation / retrieval / noise_robustness / concept_generalization
"""
import os
import json
import time
import numpy as np
from scipy.sparse import load_npz
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROTOCOL = os.path.join(HERE, "protocol.json")


def load_protocol(path=None):
    path = path or os.environ.get("FLY_PROTOCOL", DEFAULT_PROTOCOL)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def positivize(vec):
    """若非零支持集 >90% 为负，整体取负（发放率语义翻正）。cos/检索指标不变。"""
    nz = vec[vec != 0]
    if nz.size > 0 and (nz < 0).mean() > 0.9:
        return -vec
    return vec


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na * nb > 0 else 0.0


def _jaccard(a, b, th=0.1):
    sa = np.abs(a) > th
    sb = np.abs(b) > th
    u = (sa | sb).sum()
    return float((sa & sb).sum() / u) if u else 0.0


class EvalContext:
    """固定协议上下文：网络、KC、以及全部固定种子的模式集。"""

    def __init__(self, protocol=None, net_path=None):
        self.protocol = protocol or load_protocol()
        p = self.protocol
        env = p["network"].get("env_override", "FLY_NET")
        path = net_path or os.environ.get(env, p["network"]["path"])
        self.W = load_npz(path).tocsr()
        self.N = self.W.shape[0]
        # NREM 扩展：逐突触增益 g 支持。W_eff = W * g（按边），默认 g=None 即原始网络。
        self._W_base = self.W
        self._W_eff = self.W
        self._g = None
        din = np.asarray(self.W.getnnz(axis=0)).ravel()
        self.KC = np.where(din >= np.percentile(din, 99))[0]
        self.pool = np.setdiff1d(np.arange(self.N), self.KC)

        s = p["simulation"]
        self.gain = float(s["gain"])
        self.alpha = float(s["alpha"])
        self.steps = int(s["steps"])
        self.tstim = int(s["stim_steps"])
        self.amp = float(s["amp"])
        self.th = float(p["readout"]["active_threshold"])
        self.kw_k = float(p["readout"]["kwta_k_percent"])

        self._build_patterns()
        self.RP = (np.random.default_rng(7).standard_normal((len(self.KC), self.N), dtype=np.float32)) / np.sqrt(400.0)

    # ---------- 模式生成（与 phase4/5、mb_sparse 完全同种子） ----------
    def rand_pattern(self, rng, k=400):
        return rng.choice(self.pool, k, replace=False)

    def scrambler(self, base, frac, rng):
        n = int(round(len(base) * frac))
        if n <= 0:
            return base.copy()
        drop = rng.choice(base, n, replace=False)
        mask = np.ones(len(self.pool), dtype=bool)
        mask[np.searchsorted(self.pool, base)] = False
        rem = self.pool[mask]
        add = rng.choice(rem, n, replace=False)
        return np.concatenate([np.setdiff1d(base, drop), add])

    def _build_patterns(self):
        p = self.protocol["patterns"]
        self.PAIRS = []
        rng = np.random.default_rng(p["pairs"]["seed"])
        for R in p["pairs"]["levels"]:
            for _ in range(p["pairs"]["per_level"]):
                base = self.rand_pattern(rng)
                part = self.scrambler(base, R, rng)
                self.PAIRS.append((R, base, part))

        rng2 = np.random.default_rng(p["bank"]["seed"])
        self.BANK = [self.rand_pattern(rng2) for _ in range(p["bank"]["size"])]
        qsel = rng2.choice(p["bank"]["size"], p["queries"]["per_noise"], replace=False)
        self.QUERIES = []
        for noise in p["queries"]["noise_levels"]:
            for bidx in qsel:
                qp = self.scrambler(self.BANK[int(bidx)], noise, rng2)
                self.QUERIES.append((noise, int(bidx), qp))

        rng3 = np.random.default_rng(self.protocol["stability"]["pattern_seed"])
        stab_pool = [self.rand_pattern(rng3) for _ in range(6)]
        self.STAB_PATTERNS = stab_pool[: self.protocol["stability"]["n_patterns"]]

        cg = self.protocol["concept_generalization"]
        rng4 = np.random.default_rng(cg["seed"])
        self.CONCEPTS = []
        for _ in range(cg["n_concepts"]):
            base = self.rand_pattern(rng4)
            train = [self.scrambler(base, cg["train_scramble"], rng4) for _ in range(cg["train_variants"])]
            test = [self.scrambler(base, cg["train_scramble"], rng4) for _ in range(cg["test_variants"])]
            self.CONCEPTS.append((base, train, test))

    # ---------- 基础模拟 ----------
    def kwta(self, v, kpct=None):
        m = len(v)
        k = max(1, int(np.ceil(m * (kpct or self.kw_k) / 100.0)))
        if k >= m:
            return v.copy()
        idx = np.argpartition(np.abs(v), m - k)[m - k:]
        o = np.zeros(m, np.float32)
        o[idx] = v[idx]
        return o

    def _kwta_inplace_kc(self, s):
        sub = s[self.KC]
        m = len(sub)
        k = max(1, int(np.ceil(m * self.kw_k / 100.0)))
        if k < m:
            keep = np.argpartition(np.abs(sub), m - k)[m - k:]
            tmp = np.zeros(m, np.float32)
            tmp[keep] = sub[keep]
            s[self.KC] = tmp

    def sim_base(self, x, noise=0.0, seed=None, kwta_stepped=False):
        rng = np.random.default_rng(seed) if noise else None
        s = np.zeros(self.N, np.float32)
        for t in range(self.steps):
            d = self.gain * (self._W_eff @ s) + (x if t < self.tstim else 0.0)
            if noise:
                d = d + (noise * rng.standard_normal(self.N)).astype(np.float32)
            s = (1 - self.alpha) * s + self.alpha * np.tanh(d)
            if kwta_stepped:
                self._kwta_inplace_kc(s)
        return s

    def sim_apl(self, x, tau, inh, noise=0.0, seed=None):
        rng = np.random.default_rng(seed) if noise else None
        s = np.zeros(self.N, np.float32)
        apl = 0.0
        for t in range(self.steps):
            apl = apl * (1 - 1 / tau) + (1 / tau) * float(np.mean(np.maximum(s[self.KC], 0.0)))
            d = self.gain * (self._W_eff @ s) + (x if t < self.tstim else 0.0)
            d[self.KC] -= inh * apl
            if noise:
                d = d + (noise * rng.standard_normal(self.N)).astype(np.float32)
            s = (1 - self.alpha) * s + self.alpha * np.tanh(d)
        return s

    def stim_vec(self, stim):
        x = np.zeros(self.N, np.float32)
        x[stim] = self.amp
        return x

    # ---------- NREM 扩展：逐突触增益 g ----------
    def apply_gain(self, g):
        """g: 形状 (W.nnz,) 的逐突触增益（W_eff = W * g，按边）；None = 恢复原始网络。"""
        if g is None:
            self._W_eff = self._W_base
            self._g = None
        else:
            g = np.asarray(g, dtype=np.float32)
            if g.shape[0] != self._W_base.nnz:
                raise ValueError(f'g 长度 {g.shape[0]} != nnz {self._W_base.nnz}')
            We = self._W_base.copy()
            We.data = (self._W_base.data * g).astype(np.float32)
            self._W_eff = We
            self._g = g

    def clear_gain(self):
        self.apply_gain(None)

    def sim_step(self, s, drive_x=None, noise=0.0, rng=None):
        """单步推进（供连续长轨道使用；与 sim_base 的单步语义一致）。"""
        d = self.gain * (self._W_eff @ s)
        if drive_x is not None:
            d = d + drive_x
        if noise:
            if rng is None:
                rng = np.random.default_rng(0)
            d = d + (noise * rng.standard_normal(self.N)).astype(np.float32)
        return (1 - self.alpha) * s + self.alpha * np.tanh(d)


# ---------- 内置编码器（历史基线 / 参考实现） ----------
def make_plain(ctx):
    def enc(stim, noise=0.0, seed=None):
        return ctx.sim_base(ctx.stim_vec(stim), noise=noise, seed=seed)[ctx.KC].copy()
    return enc


def make_kwta_readout(ctx):
    """phase4/5 口径：sim_base + 读出端 kWTA 一次。"""
    def enc(stim, noise=0.0, seed=None):
        return ctx.kwta(ctx.sim_base(ctx.stim_vec(stim), noise=noise, seed=seed)[ctx.KC])
    return enc


def make_kwta_stepped(ctx):
    """mb_sparse 口径：每步对 KC 应用 kWTA（稳定性历史基线）。"""
    def enc(stim, noise=0.0, seed=None):
        return ctx.sim_base(ctx.stim_vec(stim), noise=noise, seed=seed, kwta_stepped=True)[ctx.KC].copy()
    return enc


def make_rp(ctx):
    def enc(stim, noise=0.0, seed=None):
        return ctx.kwta(ctx.RP @ ctx.stim_vec(stim))
    return enc


def make_apl(ctx, tau=5.0, inh=3000.0):
    def enc(stim, noise=0.0, seed=None):
        return positivize(ctx.sim_apl(ctx.stim_vec(stim), tau, inh, noise=noise, seed=seed)[ctx.KC].copy())
    return enc


def make_hybrid(ctx, tau=5.0, inh=100000.0):
    """APL + 读出 kWTA + 极性翻正（晴菜批准的符号约定）。"""
    def enc(stim, noise=0.0, seed=None):
        return positivize(ctx.kwta(ctx.sim_apl(ctx.stim_vec(stim), tau, inh, noise=noise, seed=seed)[ctx.KC]))
    return enc


# ---------- 四项评测 ----------
def eval_distance(ctx, enc):
    t0 = time.time()
    cb = [enc(b) for _, b, _ in ctx.PAIRS]
    cp = [enc(p) for _, _, p in ctx.PAIRS]
    xs = 1 - np.array([R for R, _, _ in ctx.PAIRS])
    rc = [_cos(cb[i], cp[i]) for i in range(len(ctx.PAIRS))]
    rho_s = float(spearmanr(xs, rc)[0])
    cbR = [np.maximum(c, 0) for c in cb]
    cpR = [np.maximum(c, 0) for c in cp]
    rc_r = [_cos(cbR[i], cpR[i]) for i in range(len(ctx.PAIRS))]
    try:
        rho_r = float(spearmanr(xs, rc_r)[0])
    except Exception:
        rho_r = float("nan")
    floor = float(np.mean([_cos(cb[i], cb[j]) for i in range(10) for j in range(i + 1, 10)]))
    allc = np.stack(cb + cp)
    sparse = float(np.mean(np.abs(allc) > ctx.th))
    posf = float(np.mean(allc > ctx.th))
    levels = ctx.protocol["patterns"]["pairs"]["levels"]
    per_lvl = {}
    n_per = ctx.protocol["patterns"]["pairs"]["per_level"]
    for li, R in enumerate(levels):
        sl = rc[li * n_per:(li + 1) * n_per]
        per_lvl[str(R)] = round(float(np.mean(sl)), 4) if sl else None
    return {
        "rho_s": round(rho_s, 4), "rho_r": round(rho_r, 4), "floor_s": round(floor, 4),
        "sparse_abs": round(sparse, 4), "posf": round(posf, 4), "per_level_cos": per_lvl,
        "eval_s": round(time.time() - t0, 1),
    }


def eval_retrieval(ctx, enc):
    t0 = time.time()
    bank = [enc(p) for p in ctx.BANK]
    bankR = [np.maximum(b, 0) for b in bank]
    out = {}
    for noise in ctx.protocol["patterns"]["queries"]["noise_levels"]:
        rs, rr = [], []
        for nz, bidx, qp in ctx.QUERIES:
            if nz != noise:
                continue
            qs = enc(qp)
            qR = np.maximum(qs, 0)
            ss = np.array([_cos(qs, b) for b in bank])
            sr = np.array([_cos(qR, b) for b in bankR])
            rs.append(int(np.where(np.argsort(-ss) == bidx)[0][0]))
            rr.append(int(np.where(np.argsort(-sr) == bidx)[0][0]))
        rs = np.array(rs)
        rr = np.array(rr)
        out[f"noise_{noise}"] = {
            "t1s": round(float((rs == 0).mean()), 4), "t5s": round(float((rs < 5).mean()), 4),
            "t1r": round(float((rr == 0).mean()), 4), "t5r": round(float((rr < 5).mean()), 4),
        }
    out["eval_s"] = round(time.time() - t0, 1)
    return out


def eval_noise_robustness(ctx, enc):
    """稳定性：同一模式 + 不同噪声流（rep_seeds）的重复一致性。
    注意：该指标对噪声流敏感（见 protocol v1.1 sensitivity_note），只做区间/前后对比。"""
    t0 = time.time()
    st = ctx.protocol["stability"]
    out = {}
    for lvl in st["noise_levels"]:
        cs, js, per = [], [], []
        for pat in ctx.STAB_PATTERNS:
            reps = [enc(pat, noise=lvl, seed=s) for s in st["rep_seeds"]]
            c_pat, j_pat = [], []
            for i in range(len(reps)):
                for j in range(i + 1, len(reps)):
                    c_pat.append(_cos(reps[i], reps[j]))
                    j_pat.append(_jaccard(reps[i], reps[j]))
            cs += c_pat
            js += j_pat
            per.append({"cos": round(float(np.mean(c_pat)), 4), "jaccard": round(float(np.mean(j_pat)), 4)})
        out[f"noise_{lvl}"] = {
            "cos": round(float(np.mean(cs)), 4), "jaccard": round(float(np.mean(js)), 4),
            "cos_min": round(float(np.min(cs)), 4), "cos_max": round(float(np.max(cs)), 4),
            "per_pattern": per,
        } if cs else None
    out["eval_s"] = round(time.time() - t0, 1)
    return out


def eval_concept(ctx, enc):
    """v1 草案：prototype 最近邻概念分类（clean + 二次扰动）。"""
    t0 = time.time()
    cg = ctx.protocol["concept_generalization"]
    protos = []
    for base, train, test in ctx.CONCEPTS:
        te = np.stack([enc(v) for v in train])
        protos.append(te.mean(axis=0))
    protos = np.stack(protos)
    out = {}

    def _acc(test_set_fn):
        accs = []
        for ci, (base, train, test) in enumerate(ctx.CONCEPTS):
            for t in test_set_fn(test):
                e = enc(t)
                sims = [_cos(e, p) for p in protos]
                accs.append(int(np.argmax(sims) == ci))
        return float(np.mean(accs)) if accs else 0.0

    out["acc_clean"] = round(_acc(lambda test: test), 4)
    rng = np.random.default_rng(cg["seed"] + cg.get("scramble_rng_offset", 999))
    for lvl in cg["test_scramble"]:
        out[f"acc_noise_{lvl}"] = round(_acc(lambda test, lvl=lvl: [ctx.scrambler(t, lvl, rng) for t in test]), 4)
    out["eval_s"] = round(time.time() - t0, 1)
    return out


def evaluate_encoder(ctx, enc, name="", points=None):
    points = points or ["distance", "retrieval", "noise_robustness", "concept"]
    t0 = time.time()
    out = {
        "name": name,
        "protocol_version": ctx.protocol["version"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kc_count": int(len(ctx.KC)),
        "metrics": {},
    }
    if "distance" in points:
        out["metrics"]["distance"] = eval_distance(ctx, enc)
    if "retrieval" in points:
        out["metrics"]["retrieval"] = eval_retrieval(ctx, enc)
    if "noise_robustness" in points:
        out["metrics"]["noise_robustness"] = eval_noise_robustness(ctx, enc)
    if "concept" in points:
        out["metrics"]["concept_generalization"] = eval_concept(ctx, enc)
    out["total_s"] = round(time.time() - t0, 1)
    return out


# ---------- 自检（历史校验点回归） ----------
def selftest(quick=False):
    ctx = EvalContext()
    print(f"== eval_kit selftest == net N={ctx.N} KC={len(ctx.KC)} "
          f"pairs={len(ctx.PAIRS)} bank={len(ctx.BANK)} stab={len(ctx.STAB_PATTERNS)} concepts={len(ctx.CONCEPTS)}")
    checks = []

    enc = make_plain(ctx)
    r = eval_distance(ctx, enc)
    ok = abs(r["rho_s"] - 0.3711) < 0.01
    checks.append(("plain rho_s≈0.3711", ok, r["rho_s"]))
    print(f"[1] plain rho_s={r['rho_s']} (expect 0.3711±0.01) -> {'PASS' if ok else 'FAIL'}")

    enc = make_kwta_readout(ctx)
    r = eval_distance(ctx, enc)
    ok = 0.56 <= r["rho_s"] <= 0.70
    checks.append(("kwta_readout rho_s≈0.62~0.64", ok, r["rho_s"]))
    print(f"[2] kwta_readout rho_s={r['rho_s']} sparse={r['sparse_abs']} (expect 0.56~0.70) -> {'PASS' if ok else 'FAIL'}")

    enc = make_hybrid(ctx, tau=5, inh=100000)
    r = eval_distance(ctx, enc)
    ok = (0.74 <= r["rho_s"] <= 0.84) and (not np.isnan(r["rho_r"])) and abs(r["rho_r"] - r["rho_s"]) < 0.05 \
         and abs(r["posf"] - 0.102) < 0.02
    checks.append(("hybrid flip rho_s≈0.79, rho_r==rho_s, posf≈0.102", ok, r["rho_s"]))
    print(f"[3] hybrid_flip rho_s={r['rho_s']} rho_r={r['rho_r']} posf={r['posf']} "
          f"(expect rho≈0.79, rho_r≈rho_s, posf≈0.102) -> {'PASS' if ok else 'FAIL'}")

    enc = make_kwta_stepped(ctx)
    s = eval_noise_robustness(ctx, enc)
    det = s["noise_0.0"]["cos"]
    n05 = s["noise_0.05"]["cos"]
    ok = (abs(det - 1.0) < 0.01) and (0.0 <= n05 <= 0.6)
    checks.append(("kwta_stepped 可执行性 + det=1 + n05 合理区间", ok, n05))
    print(f"[4] kwta_stepped stab: det={det} n02={s['noise_0.02']['cos']} n05={n05} "
          f"(min~max {s['noise_0.05']['cos_min']}~{s['noise_0.05']['cos_max']}) n10={s['noise_0.1']['cos']}")
    print(f"    note: 敏感指标（噪声流抽样波动大），校验可执行性+det=1 -> {'PASS' if ok else 'FAIL'}")

    if not quick:
        enc = make_kwta_readout(ctx)
        rr = eval_retrieval(ctx, enc)
        ok = 0.7 <= rr["noise_0.1"]["t1s"] <= 1.0
        checks.append(("kwta_readout t1s@0.1", ok, rr["noise_0.1"]["t1s"]))
        print(f"[5] kwta_readout t1s@0.1={rr['noise_0.1']['t1s']} t1s@0.3={rr['noise_0.3']['t1s']} "
              f"(expect 0.7~1.0) -> {'PASS' if ok else 'FAIL'}")

    passed = all(c[1] for c in checks)
    print(f"== selftest {'ALL PASS' if passed else 'SOME FAILED'} ({len(checks)} checks) ==")
    return passed


def main():
    import sys
    args = sys.argv[1:]
    if "--selftest" in args:
        selftest(quick="--quick" in args)
        return
    if "--demo" in args:
        ctx = EvalContext()
        enc = make_kwta_readout(ctx)
        res = evaluate_encoder(ctx, enc, name="kwta_readout")
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return
    print(__doc__)


if __name__ == "__main__":
    main()