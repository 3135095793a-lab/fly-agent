#!/usr/bin/env python3
"""提取 CX 内部『航向罗盘』子网：EB(椭球体) + PB(前脑桥) + NO(小结) + BU(球)
   这是果蝇的 ring attractor —— 天生该有记忆功能的结构"""
import os, pyarrow as pa, pyarrow.compute as pc, numpy as np, json, time
from scipy.sparse import csr_matrix, save_npz

SRC='/tmp/fly/conn_full.feather'; OUT=os.environ.get('FLY_OUT', '/root/fly-agent/out')
CORE = ['EB','PB','NO','BU_L','BU_R']            # 罗盘核心
CTX  = ['LAL_L','LAL_R','CRE_L','CRE_R','IB_L','IB_R','ICL_L','ICL_R','SPS_L','SPS_R']  # 上下文
src = pa.memory_map(SRC,'r'); r = pa.ipc.open_file(src)
vset = pa.array(CORE)
cols = ['pre_pt_root_id','post_pt_root_id','syn_count','gaba_avg','ach_avg','glut_avg','neuropil']
acc = {c: [] for c in cols}
t0=time.time(); tot=0
for i in range(r.num_record_batches):
    b = r.get_batch(i)
    fb = b.filter(pc.is_in(b.column('neuropil'), value_set=vset))
    if fb.num_rows:
        for c in cols: acc[c].append(fb.column(c).to_numpy(zero_copy_only=False) if c!='neuropil' else fb.column(c).to_pylist())
        tot += fb.num_rows
print(f'罗盘核心连接: {tot:,}  ({time.time()-t0:.1f}s)', flush=True)
if tot == 0: raise SystemExit('空')
pre=np.concatenate(acc['pre_pt_root_id']); post=np.concatenate(acc['post_pt_root_id'])
syn=np.concatenate(acc['syn_count']).astype(np.float32)
gaba=np.concatenate(acc['gaba_avg']).astype(np.float32)
ach=np.concatenate(acc['ach_avg']).astype(np.float32)
glut=np.concatenate(acc['glut_avg']).astype(np.float32)
nodes=np.unique(np.concatenate([pre,post])); N=len(nodes)
print(f'神经元: {N:,}', flush=True)
inhib = gaba > (ach+glut)
sign = np.where(inhib,-1.0,1.0).astype(np.float32)
ip = np.searchsorted(nodes,pre).astype(np.int32); ipost_ = np.searchsorted(nodes,post).astype(np.int32)
W = csr_matrix((syn*sign, (ip, ipost_)), shape=(N,N))
print(f'抑制性 {inhib.mean()*100:.1f}% | nnz {W.nnz:,} | 内存 {(W.data.nbytes+W.indices.nbytes+W.indptr.nbytes)/1e6:.1f} MB', flush=True)
save_npz(f'{OUT}/compass_W.npz', W); np.save(f'{OUT}/compass_nodes.npy', nodes)

# 环结构检验：入度/出度分布 + 是否有强循环
deg_in = np.asarray(W.getnnz(axis=0)).ravel(); deg_out = np.asarray(W.getnnz(axis=1)).ravel()
print(f'入度 均值{deg_in.mean():.1f} 最大{deg_in.max()} | 出度 均值{deg_out.mean():.1f} 最大{deg_out.max()}', flush=True)
json.dump({'connections':int(tot),'neurons':int(N),'nnz':int(W.nnz),
           'inhib_pct':round(float(inhib.mean()*100),2),
           'mean_in':float(deg_in.mean()),'mean_out':float(deg_out.mean())},
          open(f'{OUT}/compass_meta.json','w'), indent=2)
print('DONE', flush=True)
