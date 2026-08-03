# 原始要求复核与 OpenDriveVLA 适配状态

复核依据：`大数据综合课程实践课题-拔尖班-2026.docx` 第 13–15 页，题目 4。

## 1. 原始验收要求映射

| 原始要求 | 当前已有 | 仍需正式实验补齐 |
|---|---|---|
| 5K～10K Trajectory→Critic→Reflection 数据 | 5K 合成机制数据、统一 schema、过滤与质量分 | 使用 OpenDriveVLA/nuScenes 真实 rollout 替换或补充，并标注真实、增强、代理来源 |
| Safety/Rule/Comfort 三维 Critic | Rule-based、离线 LLM proxy、轻量 Reward Model | 用真实 VLA 失败样本做一致性、可解释性和失败定位实验；建议抽样接入真实 LLM Critic |
| OpenDriveVLA Baseline | checkpoint 与官方推理代码已下载；输出解析器和缓存适配已实现 | 在 Linux/CUDA/nuScenes 环境运行官方基线并保存结果 |
| SFT | CPU 机制版 SFT 已实现 | 以 OpenDriveVLA checkpoint 初始化 LoRA/PEFT SFT |
| Reflection SFT | CPU 机制版 Reflection SFT 已实现 | 同一基座、同一数据划分完成 LoRA Reflection SFT |
| Reflection+DPO | 机制版已实现 | 原文为可选；算力允许时再做真实 adapter DPO |
| 复杂/未见场景与碰撞、违章、平稳性 | 合成开放环指标与未见场景切分已有 | 补充 nuScenes 官方轨迹 L2/碰撞指标，并保持代理指标不冒充闭环成绩 |
| 全套代码与可复现说明 | 数据、训练、评估、Demo、一键流程已有 | 增加 GPU 环境日志、checkpoint/adapter 版本、真实实验命令和输出 |

## 2. 已完成的真实基座适配

- 自动核验 checkpoint 必需文件、Safetensors 头、模型架构和官方代码组件。
- 已识别本地架构 `LlavaQwenForCausalLM`，而不是把文件夹存在当作加载成功。
- 支持解析官方 `<traj_start>[(x,y),...]<traj_end>` 六点、三秒轨迹。
- 支持将官方 `plan_conv.json` 与场景清单合并为 Demo 缓存。
- Web Demo 采用 `auto/cache/lite` 三种来源选择，找不到真实缓存时会显式回退。
- `/api/meta` 与 `/api/audit` 分别提供运行状态和完整资产查验，不再固定报告 `weights_loaded=false` 而忽略 checkpoint 已安装状态。

## 3. 当前不能被误报为完成的部分

- 当前 Windows Python 环境没有 PyTorch、Transformers、DeepSpeed、MMCV、MMDet3D，不能执行官方实时推理。
- 还没有 nuScenes、CAN bus、map、UniAD info、cached info 和评估 GT，不能产生官方指标。
- OpenDriveVLA 官方仓库当前未发布训练脚本，真实 LoRA SFT/Reflection SFT 需要在官方 LLaVA Trainer 基础上自行实现。
- `models/*.json` 和现有 `outputs/metrics.json` 是轻量机制实验，不是 OpenDriveVLA adapter 或 nuScenes 官方结果。

## 4. 推荐最终运行结构

```text
Linux/CUDA：OpenDriveVLA + nuScenes → plan_conv.json / LoRA adapters / official metrics
                                      ↓ 导入
Windows Demo：真实轨迹缓存 → Critic → Reflection → 重规划动画与对比
                                      ↘ 无匹配缓存时显式 LiteVLA 回退
```

该结构既保留真实基座证据，又保证课堂演示无需现场加载 CUDA 模型。
