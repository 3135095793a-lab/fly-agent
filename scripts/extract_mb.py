#!/usr/bin/env python3
"""提取蘑菇体(MB)——果蝇的学习中枢。对应 FlyModel 的来源回路"""
import os, pyarrow as pa, pyarrow.compute as pc, numpy as np, json, time
from scipy.sparse import csr_matrix, save_npz
SRC='/tmp/fly/conn_full.feather'; OUT=os.environ.get('FLY_OUT', '/root/fly-agent/out')
MB = ['MB_CA_L','MB_CA_R','MB_ML_L','MB_ML_R','MB_PED_L','MB_PED_R','MB_VL_L','MB_VL_R']
AL = ['AL_L','AL_R']              # 触角叶（嗅觉输入）
allr = MB + AL
src = pa.memory_map(SRC,'r'); r = pa.ipc.open_file(src)
vset = pa.array(allr)
cols = ['pre_pt_root_id','post_pt_root_id','syn_count','gaba_avg','ach_avg','glut_avg','neuropil']
acc = {c: [] for c in cols}
t0=time.time(); tot=0
for i in range(r.num_record_batches):
    b = r.get_batch(i)
    fb = b.filter(pc.is_in(b.column('neuropil'), value_set=vset))
    if fb.num_rows:
        for c in cols:
            acc[c].append(fb.column(c).to_numpy(zero_copy_only=False) if c!='neuropil' else fb.column(c).to_pylist())
        tot += fb.num_rows
print(f'MB+AL 连接 {tot:,}  ({time.time()-t0:.1f}s)', flush=True)
if tot == 0: raise SystemExit('空')
pre=np.concatenate(acc['pre_pt_root_id']); post=np.concatenate(acc['post_pt_root_id'])
syn=np.concatenate(acc['syn_count']).astype(np.float32)
gaba=np.concatenate(acc['gaba_avg']).astype(np.float32)
ach=np.concatenate(acc['ach_avg']).astype(np.float32)
glut=np.concatenate(acc['glut_avg']).astype(np.float32)
nodes=np.unique(np.concatenate([pre,post])); N=len(nodes)
inhib = gaba > (ach+glut)
print(f'神经元 {N:,} | 抑制性 {inhib.mean()*100:.1f}%', flush=True)
ip=np.searchsorted(nodes,pre).astype(np.int32); ip2=np.searchsorted(nodes,post).astype(np.int32)
W = csr_matrix((syn*np.where(inhib,-1,1).astype(np.float32), (ip,ip2)), shape=(N,N))
save_npz(f'{OUT}/mb_W.npz', W); np.save(f'{OUT}/mb_nodes.npy', nodes)
# 扇入扇出统计（稀疏编码的关键：KC 应该被高度收敛/发散的输入驱动）
din = np.asarray(W.getnnz(axis=0)).ravel(); dout = np.asarray(W.getnnz(axis=1)).ravel()
print(f'入度 均值{din.mean():.1f} 中位{int(np.median(din))} 最大{din.max()}', flush=True)
print(f'出度 均值{dout.mean():.1f} 中位{int(np.median(dout))} 最大{dout.max()}', flush=True)
meta={'connections':int(tot),'neurons':int(N),'nnz':int(W.nnz),'inhib_pct':round(float(inhib.mean()*100),2),
      'mean_in':float(din.mean()),'mean_out':float(dout.mean()),
      'regions':{k:int(v) for k,v in sorted(__import__('collections').Counter([x for l in acc['neuropil'] for x in l]).items(), key=lambda kv:-kv[1])}}
json.dump(meta, open(f'{OUT}/mb_meta.json','w'), ensure_ascii=False, indent=2)
print(json.dumps(meta['regions'], ensure_ascii=False), flush=True)
print('DONE', flush=True)
