#!/usr/bin/env python3
"""CI数据校验：加载 data/ 全部文件，核对形状/稀疏规模/校验和，输出 out/data_validation.json"""
import os
import json
import hashlib
import glob
import numpy as np
from scipy.sparse import load_npz

os.makedirs('out', exist_ok=True)
report = {'files': {}, 'checks': [], 'ok': True}


def fail(msg):
    report['ok'] = False
    report['checks'].append({'result': 'FAIL', 'msg': msg})
    print('[FAIL]', msg, flush=True)


def info(msg):
    report['checks'].append({'result': 'INFO', 'msg': msg})
    print('[OK]', msg, flush=True)


def sha16(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


for p in sorted(glob.glob('data/*')):
    name = os.path.basename(p)
    entry = {'sha16': sha16(p), 'bytes': os.path.getsize(p)}
    report['files'][name] = entry
    print(f'[FILE] {name} {entry["bytes"]}B sha16={entry["sha16"]}', flush=True)

expected_w = {
    'compass_W.npz': {'N': 4193, 'nnz': 83694},
    'cx_W.npz': {'N': 29474, 'nnz': 1841414},
    'mb_W.npz': {'N': 18015, 'nnz': 850678},
}

for name, exp in expected_w.items():
    try:
        W = load_npz(f'data/{name}').tocsr()
        entry = {'N': int(W.shape[0]), 'nnz': int(W.nnz),
                 'inhib_pct': round(float((W.data < 0).mean() * 100), 2)}
        report['files'][name].update(entry)
        print(f'[NPZ] {name}: N={entry["N"]} nnz={entry["nnz"]} inhib={entry["inhib_pct"]}%', flush=True)
        if entry['N'] != exp['N'] or entry['nnz'] != exp['nnz']:
            fail(f'{name} mismatch: expected N={exp["N"]} nnz={exp["nnz"]}, got N={entry["N"]} nnz={entry["nnz"]}')
        else:
            info(f'{name} 与预期一致（{exp["N"]} 神经元 / {exp["nnz"]} 突触）')
    except Exception as e:
        fail(f'{name} 加载失败: {e}')

try:
    W = load_npz('data/mb2_W.npz').tocsr()
    entry = {'N': int(W.shape[0]), 'nnz': int(W.nnz)}
    report['files']['mb2_W.npz'].update(entry)
    print(f'[NPZ] mb2_W.npz: N={entry["N"]} nnz={entry["nnz"]}', flush=True)
    info('mb2_W 加载正常（无预期基线，仅记录）')
except Exception as e:
    fail(f'mb2_W 加载失败: {e}')

expected_npy = {
    'compass_nodes.npy': 4193,
    'cx_nodes.npy': 29474,
    'cx_nodes2.npy': 29474,
    'mb_nodes.npy': 18015,
    'mb2_nodes.npy': 38192,
}
for name, exp_len in expected_npy.items():
    try:
        arr = np.load(f'data/{name}', allow_pickle=False)
        entry = {'shape': list(arr.shape), 'dtype': str(arr.dtype)}
        report['files'][name].update(entry)
        print(f'[NPY] {name}: shape={entry["shape"]} dtype={entry["dtype"]}', flush=True)
        if arr.shape and arr.shape[0] == exp_len:
            info(f'{name} 与预期长度一致（{exp_len}）')
        else:
            fail(f'{name} 长度异常 {arr.shape} != {exp_len}')
    except Exception as e:
        fail(f'{name} 加载失败: {e}')

for name in ['cx_pre.npy', 'cx_post.npy', 'cx_cnt.npy']:
    try:
        arr = np.load(f'data/{name}', allow_pickle=False)
        entry = {'shape': list(arr.shape), 'dtype': str(arr.dtype)}
        report['files'][name].update(entry)
        print(f'[NPY] {name}: shape={entry["shape"]} dtype={entry["dtype"]}', flush=True)
    except Exception as e:
        fail(f'{name} 加载失败: {e}')

json.dump(report, open('out/data_validation.json', 'w'), indent=2)
print('VALIDATION_OK' if report['ok'] else 'VALIDATION_HAS_FAILURES', flush=True)
raise SystemExit(0 if report['ok'] else 1)
