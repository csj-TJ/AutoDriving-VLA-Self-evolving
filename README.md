# 基于 Driving Reflection 与 Critic 反馈的自动驾驶 VLA 自进化

> Windows 运行 Demo 请只安装 `requirements.txt`；下载/审计基座另装
> `requirements-model.txt`。`requirements-gpu.txt` 面向 Linux/CUDA 的 OpenDriveVLA
> PEFT 环境，DeepSpeed 分布式训练依赖已单独放在 `requirements-deepspeed.txt`。

本项目是课程选题第 4 题的可复现轻量实现。它把自动驾驶失败经验转换为结构化训练知识，完整覆盖 `Trajectory → Critic → Reflection → Re-learning`，并提供 Rule-based、LLM Critic（在线接口/离线确定性代理）和 Reward Model 三种评价方式。

## 克隆后直接运行 Demo

```powershell
python -m pip install -r requirements.txt
python scripts/run_demo.py
```

浏览器打开 `http://127.0.0.1:8000`。仓库已包含从 nuScenes v1.0-mini 抽取并完成
轻量训练的 5,112 条 `data/reflection_dataset.jsonl`，因此无原始图像也能运行轨迹、评分
和反思 Demo；下载原始 nuScenes mini 后，页面还会显示对应 CAM_FRONT 实景图像。

Windows 也可直接双击 `启动Demo.bat`。新版驾驶台从完整训练数据索引动态选择真实路口、行人运动、近距前车和转弯场景，同屏比较 Baseline 与反思后轨迹，并展示模型调用、数据近邻、Critic 评分和结构化 Reflection。前车按场景速度运动，行人按 nuScenes `sample_annotation` 时间轨迹运动，自车停车后保持道路航向；后端命令行只输出简洁的人类可读运行摘要。详细操作见 `docs/Demo使用说明.md`。

驾驶台会把本轮失败证据进一步抽象为可复用驾驶 Skill，实时展示“失败捕获→反思归因→Skill 抽象→重规划验证→记忆固化”过程。Skill 的触发条件来自当前场景，验证增益来自 Baseline/反思后实时评分，记忆支持数来自 `outputs/reflection_memory.jsonl`，不是前端固定展示值。

## 基座模型与轻量化结论

完整模型实验**必须以 OpenDriveVLA-0.5B 为基座初始化**，再进行 LoRA Reflection SFT，而不是从零训练；Reflection+DPO 属于可选增强。本地 checkpoint 和官方源码现已下载。运行 `python scripts/audit_opendrivevla.py` 可核验文件、架构、依赖和真实推理缓存。Demo 默认采用 `auto` 模式：存在匹配的 OpenDriveVLA GPU 推理缓存时优先回放，否则明确切换到 LiteVLA CPU 机制模式。它不会把“权重已下载”误报为“权重已在当前进程加载”。详细边界与替换路径见 `docs/基座模型与轻量化说明.md`。

## 交付映射

- 数据资产：仓库直接提供 5,112 条 nuScenes 衍生 JSONL 及溯源清单，协作者无需执行下载、抽取或数据生成工具。
- Critic：`src/selfevolve_drive/critics.py` 实现 Safety/Rule/Comfort、多类型 Critic 与证据定位。
- Self-Evolution：`scripts/train.py` 实现 Baseline、SFT、失败样本加权 Reflection SFT 及偏好优化 DPO。
- 多轮进化：本地执行三轮 rollout→Critic→Reflection→失败记忆回放，历史写入 `outputs/evolution_history.json`，经验池写入 `outputs/reflection_memory.jsonl`。
- 完整训练接口：`scripts/train_opendrivevla_peft.py` 提供 LoRA/PEFT 训练入口，Windows 可用 `--check-only` 验证。
- 实验评估：`scripts/evaluate.py` 输出碰撞风险、违章、礼让失败、舒适性与策略误差，并标记未见场景测试集。
- Demo：`demo/index.html` + `demo/styles.css` + `demo/app.js` + `scripts/run_demo.py`，通过 Python API 动态生成双轨迹、Critic、Reflection 与自进化历史。
- 基座查验与适配：`scripts/audit_opendrivevla.py` 检查本地资产，`scripts/import_opendrivevla_outputs.py` 将官方 `plan_conv.json` 转为 Demo 可回放缓存。
- 完整报告：主交付为 `report/自动驾驶VLA自进化研究报告.tex` 与
  `report/自动驾驶VLA自进化研究报告_最新版.pdf`。`.tex` 是唯一可编辑源稿，报告生成工具
  不随主分支分发。参考论文与源码的
  本地情况见 `references/REFERENCE_MANIFEST.md`。
- GitHub 协作、基座依赖、数据集目录和 GPU 环境说明：见 `docs/GitHub协作与环境配置.md`。
- 实际推理与自进化训练的 WSL2/Linux/CUDA 安装步骤：见 `docs/真实推理与自进化训练环境.md`。

## 使用现有数据复现实验

仓库已经包含 Demo 和轻量训练所需的数据及模型参数。重新运行轻量训练与评估只需：

```powershell
python scripts/train.py
python scripts/evaluate.py
python scripts/run_demo.py
```

一次性数据下载、抽取、生成和报告构建工具不随远端分发；原始 nuScenes 文件、基座权重、
token 和机器路径也已通过 `.gitignore` 排除。当前指标来自
nuScenes mini 图像/标注/未来 ego pose 上的轻量开放环实验，不等同于 CARLA 或真实道路闭环成绩；
Windows Demo 使用已训练的 LiteVLA 参数，完整 OpenDriveVLA 视觉权重推理与 LoRA 训练仍需 Linux/CUDA。
