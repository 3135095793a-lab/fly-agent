#!/usr/bin/env python3
"""补上 APL：MB + AL + LA(侧副叶，APL 所在)"""
import os, pyarrow as pa, pyarrow.compute as pc, numpy as np, json, time
from scipy.sparse import csr_matrix, save_npz
SRC='/tmp/fly/conn_full.feather'; OUT=os.environ.get('FLY_OUT', '/root/fly-agent/out')
REG = ['MB_CA_L','MB_CA_R','MB_ML_L','MB_ML_R','MB_PED_L','MB_PED_R','MB_VL_L','MB_VL_R',
       'AL_L','AL_R','LA_L','LA_R']
src = pa.memory_map(SRC,'r'); r = pa.ipc.open_file(src)
vset = pa.array(REG)
cols = ['pre_pt_root_id','post_pt_root_id','syn_count','gaba_avg','ach_avg','glut_avg']
acc = {c: [] for c in cols}; tot=0
t0=time.time()
for i in range(r.num_record_batches):
    b = r.get_batch(i)
    fb = b.filter(pc.is_in(b.column('neuropil'), value_set=vset))
    if fb.num_rows:
        for c in cols: acc[c].append(fb.column(c).to_numpy(zero_copy_only=False))
        tot += fb.num_rows
print(f'MB+AL+LA 连接 {tot:,} ({time.time()-t0:.1f}s)', flush=True)
pre=np.concatenate(acc['pre_pt_root_id']); post=np.concatenate(acc['post_pt_root_id'])
syn=np.concatenate(acc['syn_count']).astype(np.float32)
gaba=np.concatenate(acc['gaba_avg']).astype(np.float32)
ach=np.concatenate(acc['ach_avg']).astype(np.float32)
glut=np.concatenate(acc['glut_avg']).astype(np.float32)
nodes=np.unique(np.concatenate([pre,post])); N=len(nodes)
inhib = gaba > (ach+glut)
print(f'神经元 {N:,} | 抑制性边 {inhib.mean()*100:.1f}%', flush=True)
ip=np.searchsorted(nodes,pre).astype(np.int32); ip2=np.searchsorted(nodes,post).astype(np.int32)
W = csr_matrix((syn*np.where(inhib,-1,1).astype(np.float32), (ip,ip2)), shape=(N,N))
save_npz(f'{OUT}/mb2_W.npz', W); np.save(f'{OUT}/mb2_nodes.npy', nodes)
print(f'nnz {W.nnz:,} | 内存 {(W.data.nbytes+W.indices.nbytes+W.indptr.nbytes)/1e6:.1f} MB', flush=True)
print('DONE', flush=True)
