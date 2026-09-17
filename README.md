# fly-agent（FlyWire 果蝇电路实验仓库）

用真实果蝇连接组（FlyWire）子电路做工作记忆任务的训练实验。正式训练跑在 GitHub Actions 上（手机 CPU 太慢）。

- 发起/实验设计：晴菜（DSH）
- 执行/CI 运维：Operit

## 目录结构

- `data/`：4 个子电路提取结果（compass 4,193；cx 29,474；mb 18,015；mb2 38,192 神经元）+ 节点表
- `scripts/`：原实验脚本。相对原版**仅改 1 行**：输出目录支持 `FLY_OUT` 环境变量（默认值与原版完全相同，本地环境不受影响）
- `ci/`：CI 工具脚本（数据校验、训练单步基准）
- `.github/workflows/`：Actions 工作流
  - `bench.yml`：基准（数据校验 + 评估链 + 训练单步测速）
  - 训练工作流：train.yml（300步 + 线程对比 + 每50步checkpoint + 训练后评估）

## 运行约定

- 脚本默认输出目录：`/root/fly-agent/out`（原环境不变）
- CI 中通过 `FLY_OUT=<dir>` 覆盖；并先把 `data/` 内容复制到该目录（脚本从那里读输入、写输出）

## 数据来源

FlyWire Consortium 公开连接组数据（https://flywire.ai）。本仓库仅用于科研实验；数据使用请遵循 FlyWire 原始条款并署名。
