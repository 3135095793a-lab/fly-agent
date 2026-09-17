#!/usr/bin/env python3
"""CX 子电路 LIF 脉冲仿真器 —— 看它能跑多快"""
import numpy as np, time, json, os
from scipy.sparse import csr_matrix

OUT = os.environ.get('FLY_OUT', '/root/fly-agent/out')
pre = np.load(f'{OUT}/cx_pre.npy'); post = np.load(f'{OUT}/cx_post.npy')
cnt = np.load(f'{OUT}/cx_cnt.npy'); nodes = np.load(f'{OUT}/cx_nodes.npy')
N = len(nodes)
t0 = time.time()
ipre = np.searchsorted(nodes, pre).astype(np.int32)
ipost = np.searchsorted(nodes, post).astype(np.int32)
del pre, post
W = csr_matrix((cnt.astype(np.float32), (ipre, ipost)), shape=(N, N))
del ipre, ipost, cnt
print(f'矩阵: {N} 神经元, {W.nnz:,} 突触, 构建 {time.time()-t0:.1f}s', flush=True)
mb = (W.data.nbytes + W.indices.nbytes + W.indptr.nbytes)/1e6
print(f'CSR 内存: {mb:.0f} MB', flush=True)

# LIF 参数
tau_m, dt, V_th, V_reset = 20.0, 1.0, 1.0, 0.0
decay = float(np.exp(-dt/tau_m))
gain = 0.02          # 突触增益
v = np.zeros(N, dtype=np.float32)
spk = np.zeros(N, dtype=np.float32)

# 刺激：随机 5% 神经元接收持续输入
rng = np.random.default_rng(42)
stim_mask = rng.random(N) < 0.05
stim_amp = 0.06

# 预热一次（编译/scipy 缓存）
I = W @ spk; v = v * decay + I * gain
t0 = time.time(); STEPS = 500
fire_total = 0
for s in range(STEPS):
    I = W @ spk                       # 突触输入（核心计算）
    v *= decay
    v += I * gain
    v[stim_mask] += stim_amp          # 外部刺激
    fired = v >= V_th
    nf = int(fired.sum())
    fire_total += nf
    spk = fired.astype(np.float32)
    v[fired] = V_reset
el = time.time() - t0
per = el / STEPS
print(f'--- {STEPS} 步仿真 ---', flush=True)
print(f'总耗时      : {el:.2f} s', flush=True)
print(f'每步        : {per*1000:.2f} ms', flush=True)
print(f'速度        : {1/per:.0f} 步/秒  ({1/per*1000:.0f} 步/秒 ≈ {1/per:.0f}x 实时@1ms)', flush=True)
print(f'平均发放    : {fire_total/STEPS:.0f} 神经元/步 ({fire_total/STEPS/N*100:.2f}%)', flush=True)
res = {'neurons': int(N), 'synapses': int(W.nnz), 'csr_mb': round(mb,1),
       'ms_per_step': round(per*1000,3), 'steps_per_sec': round(1/per,1),
       'mean_firing': round(fire_total/STEPS,1), 'firing_pct': round(fire_total/STEPS/N*100,3)}
json.dump(res, open(f'{OUT}/sim_bench.json','w'), indent=2)
print('DONE', flush=True)
