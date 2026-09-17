#!/usr/bin/env python3
"""CX 子电路 v2 —— 用递质数据区分兴奋/抑制，实现 E/I 平衡"""
import pyarrow as pa, pyarrow.compute as pc, numpy as np, time, json, os
from scipy.sparse import csr_matrix

OUT = os.environ.get('FLY_OUT', '/root/fly-agent/out')
CX = ['EB','FB','PB','NO','BU_L','BU_R','LAL_L','LAL_R','CRE_L','CRE_R','IPS_L','IPS_R',
      'SPS_L','SPS_R','IB_L','IB_R','ICL_L','ICL_R','ATL_L','ATL_R','CAN_L','CAN_R',
      'FLA_L','FLA_R','GOR_L','GOR_R','VES_L','VES_R','WED_L','WED_R','EPA_L','EPA_R']
src = pa.memory_map('/tmp/fly/conn_full.feather','r'); r = pa.ipc.open_file(src)
vset = pa.array(CX)
cols = ['pre_pt_root_id','post_pt_root_id','syn_count','gaba_avg','ach_avg','glut_avg','oct_avg','ser_avg','da_avg']
acc = {c: [] for c in cols}
t0=time.time()
for i in range(r.num_record_batches):
    b = r.get_batch(i).select(cols)
    fb = b.filter(pc.is_in(r.get_batch(i).column('neuropil'), value_set=vset))
    if fb.num_rows:
        for c in cols: acc[c].append(fb.column(c).to_numpy(zero_copy_only=False))
pre = np.concatenate(acc['pre_pt_root_id']); post = np.concatenate(acc['post_pt_root_id'])
syn = np.concatenate(acc['syn_count']).astype(np.float32)
gaba = np.concatenate(acc['gaba_avg']).astype(np.float32)
ach  = np.concatenate(acc['ach_avg']).astype(np.float32)
glut = np.concatenate(acc['glut_avg']).astype(np.float32)
print(f'提取 {len(pre):,} 连接  {time.time()-t0:.1f}s', flush=True)
nodes = np.unique(np.concatenate([pre, post])); N = len(nodes)
# E/I 判定：抑制性 = GABA 占优
inhib = gaba > (ach + glut)
sign = np.where(inhib, -1.0, 1.0).astype(np.float32)
w = syn * sign
print(f'神经元 {N:,} | 抑制性连接占比 {inhib.mean()*100:.1f}%', flush=True)
ipre = np.searchsorted(nodes, pre).astype(np.int32)
ipost = np.searchsorted(nodes, post).astype(np.int32)
W = csr_matrix((w, (ipre, ipost)), shape=(N, N))
np.save(f'{OUT}/cx_nodes2.npy', nodes)
from scipy.sparse import save_npz
save_npz(f'{OUT}/cx_W.npz', W.tocsr())
print(f'CSR: nnz={W.nnz:,}  内存 {(W.data.nbytes+W.indices.nbytes+W.indptr.nbytes)/1e6:.0f} MB', flush=True)

# LIF with E/I balance
tau_m, dt = 20.0, 1.0
decay = float(np.exp(-dt/tau_m)); gain = 0.004
V_th, V_reset = 1.0, 0.0
v = np.zeros(N, dtype=np.float32); spk = np.zeros(N, dtype=np.float32)
rng = np.random.default_rng(1)
stim_mask = rng.random(N) < 0.02
I = W @ spk; v = v*decay + I*gain
t0=time.time(); STEPS=500; tot=0
for s in range(STEPS):
    I = W @ spk
    v *= decay; v += I*gain
    v[stim_mask] += 0.05
    fired = v >= V_th
    nf = int(fired.sum()); tot += nf
    spk = fired.astype(np.float32)
    v[fired] = V_reset
el = time.time()-t0; per = el/STEPS
print(f'--- E/I 平衡后 ---', flush=True)
print(f'每步 {per*1000:.2f} ms | {1/per:.0f} 步/秒', flush=True)
print(f'发放率 {tot/STEPS/N*100:.2f}%  ({tot/STEPS:.0f} 个/步)', flush=True)
json.dump({'neurons':int(N),'nnz':int(W.nnz),'ms_per_step':round(per*1000,3),
           'steps_per_sec':round(1/per,1),'firing_pct':round(tot/STEPS/N*100,3),
           'inhib_frac':round(float(inhib.mean()),4)}, open(f'{OUT}/v2_bench.json','w'), indent=2)
print('DONE', flush=True)
