#!/usr/bin/env python3
"""从 FlyWire 连接组提取中央复合体(CX)子电路 —— 果蝇决策中枢"""
import pyarrow as pa, pyarrow.compute as pc, numpy as np, time, os, json, sys

SRC = '/tmp/fly/conn_full.feather'
OUT = os.environ.get('FLY_OUT', '/root/fly-agent/out')
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.environ.get('FLY_LOG', '/root/fly-agent/logs'), exist_ok=True)

CX = ['EB','FB','PB','NO','BU_L','BU_R','LAL_L','LAL_R','CRE_L','CRE_R','IPS_L','IPS_R',
      'SPS_L','SPS_R','IB_L','IB_R','ICL_L','ICL_R','ATL_L','ATL_R','CAN_L','CAN_R',
      'FLA_L','FLA_R','GOR_L','GOR_R','VES_L','VES_R','WED_L','WED_R','EPA_L','EPA_R']

t0 = time.time()
if not os.path.exists(SRC):
    print(f'FATAL: {SRC} 不存在'); sys.exit(1)

src = pa.memory_map(SRC, 'r')
r = pa.ipc.open_file(src)
nb = r.num_record_batches
vset = pa.array(CX)
pre_l, post_l, cnt_l, npl_l = [], [], [], []
tot = 0
for i in range(nb):
    b = r.get_batch(i)
    fb = b.filter(pc.is_in(b.column('neuropil'), value_set=vset))
    n = fb.num_rows
    if n:
        pre_l.append(fb.column('pre_pt_root_id').to_numpy())
        post_l.append(fb.column('post_pt_root_id').to_numpy())
        cnt_l.append(fb.column('syn_count').to_numpy())
        npl_l.append(fb.column('neuropil').to_pylist())
        tot += n
    if i % 40 == 0:
        print(f'batch {i}/{nb} kept={tot} t={time.time()-t0:.0f}s', flush=True)

print(f'CX 内部连接数: {tot}', flush=True)
if tot == 0:
    print('未筛到 CX 连接'); sys.exit(2)

pre = np.concatenate(pre_l); post = np.concatenate(post_l); cnt = np.concatenate(cnt_l)
nodes = np.unique(np.concatenate([pre, post]))
np.save(f'{OUT}/cx_pre.npy', pre); np.save(f'{OUT}/cx_post.npy', post)
np.save(f'{OUT}/cx_cnt.npy', cnt); np.save(f'{OUT}/cx_nodes.npy', nodes)

# 按脑区统计
from collections import Counter
c = Counter()
for lst in npl_l: c.update(lst)
meta = {
    'connections': int(tot), 'neurons': int(len(nodes)),
    'mean_syn': float(cnt.mean()), 'max_syn': int(cnt.max()),
    'regions': dict(c.most_common(20)), 'elapsed_s': round(time.time()-t0, 1),
}
with open(f'{OUT}/cx_meta.json','w') as f: json.dump(meta, f, ensure_ascii=False, indent=2)
print(json.dumps(meta, ensure_ascii=False, indent=2)[:1200], flush=True)
print('DONE', flush=True)
