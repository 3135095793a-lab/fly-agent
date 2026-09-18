# 果蝇神经 agent · 基础版（fly-agent-basic）

> 从 FlyWire 果蝇连接组实验（2026-09）沉淀的**可用版本**
> 全部指标经 GitHub Actions 多轮验证（来源见"已验证指标"）

---

## 这是什么

一个跑在**真实果蝇蘑菇体连接组**（38192 神经元）上的微型记忆 agent。核心能力：

| 能力 | 接口 | 说明 |
|---|---|---|
| 存记忆 | remember(key, data) | 支持多段记忆 parts=4（分解） |
| 取记忆 | recall(query) | 余弦检索 |
| 部分线索 | recall_blocked(blocks) | 多段记忆的部分块召回（联想） |
| 创新组合 | recombine(new, k1, k2) | 两个记忆的前后半拼成新记忆 |
| 分辨异样 | novelty(query) | 判断输入"见过没有" → (is_novel, max_sim) |
| 断连/接回 | disconnect/reconnect(key) | 零误伤的段级隔离门控（mode='hard'/'soft'） |
| 睡眠 | sleep(rounds, noise_sigma=2.0, struct_update=False) | v2：重标定 + 重随机化 |
| 存档 | save(path) / load(path) | |

## 快速开始

    cd fly-agent-basic
    python3 demo.py          # 全能力演示（约 2 分钟）
    python3 fly_agent.py     # 冒烟测试（两模式）

依赖：Python 3.10+ / numpy / scipy
数据：data/mb2_W.npz（果蝇蘑菇体子电路：38192 神经元 / 961085 突触）

## 已验证指标（来源：GitHub Actions，2026-09-18）

| 指标 | 数值 | 来源 run |
|---|---|---|
| 干净检索（100 记忆） | 1.0000 | 35313462076 |
| 加噪 0.3 检索 | 0.99 | 35308030114 |
| 断连零误伤 | 0/50 与 50/50 | 35313462076 |
| 分解（25% 输入块） | 25/25 | 35313462076 |
| 联想（50% 线索） | 0.91 | 35313462076 |
| 异样检测 | 1.00 / 1.00 | 35313462076 |
| 睡眠后检索（σ=2） | 0.98~1.00 | 35308030114 |

## 关键设计（为什么这样配置）

1. **隔离版（segmented）**：38192 神经元打乱分 100 段（每段 381），每个记忆专属一段
   → 物理互斥、断连零误伤（实测 0/50 与 50/50）
2. **睡眠用 v2**：实验证明"结构性学习"（共激活强化）对检索无益甚至有害——
   改为"增益重标定（λ^轮数）+ 重随机化（σ）"，更快且指标更高
3. **码格式**：段内 |s| top-76（20% 稀疏），余弦检索
4. **推荐配置**：mode='segmented'；sleep(20, lam=0.90, noise_sigma=2.0, struct_update=False)

## 文件结构

    fly-agent-basic/
    ├── fly_agent.py          # 核心（FlyAgent 类）
    ├── demo.py               # 全能力演示
    ├── README.md             # 本文件
    ├── dream/
    │   ├── eval_kit.py       # 网络仿真与编码（EvalContext）
    │   ├── nrem_v1.py        # 睡眠机制
    │   └── protocol.json     # 仿真协议参数
    └── data/
        └── mb2_W.npz         # 蘑菇体连接组子电路

## 来源与进一步实验

完整实验档案（含全部 CI 结果、设计文档、聊天记录）见同级目录 fly-agent-archive/。
核心文档：START-HERE.md（项目交接）与 results/agent_v2/README.md（最新实验汇总）。

