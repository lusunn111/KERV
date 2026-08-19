# KERV-FlagOS

**面向具身智能物理反馈的推测执行与 FlagOS 系统优化。**

KERV 将轻量 Drafter、主模型批量验证、动态接受机制与 Kalman/运动学反馈
组织为一条闭环 VLA 推理链。本仓库提供 KERV 在 FlagScale 上的精简开源
实现，以及面向单 Batch、短序列验证树负载的 FlagOS 专项算子和无损系统优化。

> 当前为有限研究发布：代码、最终 BF16 安全配置和训练方法已公开；模型权重、
> 生成数据、实验日志、Profiler、消融配置和论文图片不在仓库中。

## 主要特性

- **FlagScale 一键纳管：** 一份配置管理模型路径、进程、环境、日志和 LIBERO
  任务，正式运行只有一个公开入口 `run_kerv.py`。
- **具身专项算子：** 提供 18 个 `torch.ops.flagos_embodied` 接口；BF16 安全
  配置注册 14 个主线接口，其余实现以安全或实验开关保留。
- **模型计算融合：** 对实际输入规模选择性启用 QKV 与
  Gate-Up-SwiGLU 融合，未达到收益门限的输入自动走原生 CUDA 路径。
- **闭环系统优化：** 静态验证树、常驻 K/V、CUDA Graph、持久化工作区、
  CPU-GPU 控制量压缩和 Commit/Draft 双流协同均包含在最终链路中。
- **精度优先：** 默认关闭 W8A16、紧凑树及未通过完整验收的实验优化，保留
  KERV 原始阈值、Kalman 逻辑和动作语义。

## 发布范围

| 已公开 | 未公开 |
|---|---|
| 单一 FlagScale 启动入口与最终配置 | Base/Drafter 权重 |
| KERV 推理核心与 LIBERO Goal 链路 | 生成的训练样本与数据集 |
| 18 个具身算子接口及原生回退 | Profiler、微基准和内部测试脚本 |
| QKV、Gate-Up-SwiGLU 等融合实现 | 历史配置、失败实验和消融配置 |
| 静态树、常驻缓存与计算图运行时 | 实验日志、视频和论文图片 |
| Drafter 数据生成和训练方法 | 第三方模型与仿真资产 |

仓库结构：

```text
KERV-FlagOS/
├── run_kerv.py                 # 唯一公开运行入口
├── configs/kerv_libero_goal.yaml
├── kerv_flagos/                # FlagOS 算子、融合与 FlagScale 入口
├── openvla/                    # KERV/OpenVLA 最小运行依赖
├── training/                   # Drafter 数据生成与训练配置
└── docs/                       # 算子和训练说明
```

## 计算流程

```text
Observation + Instruction
          │
          ▼
   OpenVLA visual/prompt prefill
          │
          ▼
  Drafter candidate generation
          │
          ▼
 Static-tree batched verification ──► Verify / Accept
          │                                │
          └──── Kalman & kinematic feedback
                                           │
                                           ▼
                                     Robot action
```

FlagScale 负责启动、配置和日志纳管；KERV 的 Transformer 融合、静态树算子、
常驻缓存和计算图由本仓库的 `kerv_flagos` 扩展实现。详细接口见
[docs/OPERATORS.md](docs/OPERATORS.md)。

## 安装

参考环境为 Linux、Python 3.10、CUDA 12.x 和 A100。建议使用独立环境：

```bash
conda create -n kerv-flagos python=3.10 -y
conda activate kerv-flagos

git clone https://github.com/flagos-ai/FlagScale.git third_party/FlagScale
python -m pip install -e third_party/FlagScale

# 请先按本机 CUDA 版本安装 PyTorch，再安装其余依赖。
python -m pip install -r requirements.txt
python -m pip install -e openvla

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
python -m pip install -e third_party/LIBERO
```

OpenVLA 的 FlashAttention 安装方式与 CUDA/PyTorch 组合相关；如需启用，请在
PyTorch 安装完成后执行：

```bash
python -m pip install flash-attn --no-build-isolation
```

## 准备模型

权重不随仓库发布。请将合法获取或自行训练的权重整理为：

```text
checkpoints/
├── openvla-libero-goal/
│   ├── config.json
│   ├── dataset_statistics.json
│   ├── tokenizer.json
│   ├── preprocessor_config.json
│   └── model-*.safetensors
└── kerv-drafter/
    ├── config.json
    └── pytorch_model.bin
```

Base checkpoint 是在 LIBERO Goal 上适配的 OpenVLA；Drafter 可以按照
[docs/TRAINING.md](docs/TRAINING.md) 生成监督数据并训练。

## 快速运行

先执行不加载权重的配置检查：

```bash
python run_kerv.py \
  --flagscale-root third_party/FlagScale \
  --dry-run
```

运行一个完整 LIBERO Goal episode：

```bash
python run_kerv.py \
  --flagscale-root third_party/FlagScale \
  --base-checkpoint checkpoints/openvla-libero-goal \
  --draft-checkpoint checkpoints/kerv-drafter \
  --libero-config ~/.libero \
  --device 0
```

快速检查八个动作步：

```bash
python run_kerv.py \
  --flagscale-root third_party/FlagScale \
  --base-checkpoint checkpoints/openvla-libero-goal \
  --draft-checkpoint checkpoints/kerv-drafter \
  --libero-config ~/.libero \
  --max-episode-steps 8
```

默认以前台方式输出 FlagScale 与 KERV 日志；加 `--background` 可交由
FlagScale 后台管理。结果写入 `outputs/kerv_libero_goal/`。

## 默认优化配置

默认配置是 A100 BF16 安全路径：

- 48 路静态验证树，节点按 `224/240/248/256/264/280/320` 分桶；
- QKV 与 Gate-Up-SwiGLU 按实测输入规模选择性融合；
- RoPE/KV 写入、静态树 Attention 与接受路径 Commit 使用 `auto` 路由；
- Prompt、Draft、Verifier CUDA Graph 与共享内存池开启；
- 常驻主模型/Draft KV、持久化输入和控制缓冲开启；
- W8A16、紧凑树、动作头裁剪和实验 GEMM epilogue 关闭。

内部参考环境的稳态单步时延约为 **131 ms**（A100、BF16、batch=1；不含
首次编译和计算图捕获）。该数字用于说明参考配置，不代表不同驱动、权重、
任务或硬件上的保证；请同时报告冷启动、Mean、Median 和 P95。

## 训练 Drafter

训练分为两步：首先冻结 OpenVLA，生成 verifier hidden state 与 action-token
监督；随后使用 DeepSpeed ZeRO-2 训练一层 Drafter。完整命令、损失定义和
checkpoint 导出方式见 [docs/TRAINING.md](docs/TRAINING.md)。训练权重和中间
样本均由 `.gitignore` 排除。

## 正确性与复现原则

系统优化不修改候选宽度、接受阈值、Kalman 反馈或模型精度。启用新优化时，
应逐步比较：Drafter 候选顺序、动作 Token、接受长度、最佳路径、Verify 次数、
EOS、Kalman 分支和最终环境动作。首次 Triton 编译与 CUDA Graph 捕获应单独
统计，不能并入稳态加速结果。

## 已知限制

- 当前公开默认配置主要在 A100、batch=1、BF16 和 LIBERO Goal 上验收；
- FlagOS 提供跨芯片接口与编译基础，但本仓库不宣称尚未实测芯片的性能；
- 权重和 LIBERO 资产需要用户自行准备；
- 当前版本为研究代码，不建议直接用于安全关键的真实机器人控制。

## 致谢

本项目基于 OpenVLA、LIBERO、PyTorch、Triton、Hugging Face Transformers、
FlagScale 和 FlagGems。感谢相关开源社区。具体许可说明见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 许可与引用

仓库包含 MIT 与 Apache-2.0 许可的组件，详见 `LICENSE`、`licenses/` 和各源
文件头。模型与数据集遵循其各自许可。论文图片与正式 BibTeX 将在论文发布
版本确定后补充。
