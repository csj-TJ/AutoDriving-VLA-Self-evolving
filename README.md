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

浏览器打开 `http://127.0.0.1:8000`。仓库已包含 Demo 使用的 5,000 条
`data/reflection_dataset.jsonl`，无需先生成数据。需要从头复现实验时运行
`python scripts/run_all.py --samples 5000`；仅快速验证流水线时可用 `--samples 200`。

Windows 也可直接双击 `启动Demo.bat`。新版驾驶台从完整训练数据索引动态选择红灯、行人、慢速前车和恶劣弯道场景，同屏比较 Baseline 与反思后轨迹，并展示模型调用、数据近邻、Critic 评分、结构化 Reflection 和后端实时日志。详细操作见 `docs/Demo使用说明.md`。

## 基座模型与轻量化结论

完整模型实验**必须以 OpenDriveVLA-0.5B 为基座初始化**，再进行 LoRA Reflection SFT，而不是从零训练；Reflection+DPO 属于可选增强。本地 checkpoint 和官方源码现已下载。运行 `python scripts/audit_opendrivevla.py` 可核验文件、架构、依赖和真实推理缓存。Demo 默认采用 `auto` 模式：存在匹配的 OpenDriveVLA GPU 推理缓存时优先回放，否则明确切换到 LiteVLA CPU 机制模式。它不会把“权重已下载”误报为“权重已在当前进程加载”。详细边界与替换路径见 `docs/基座模型与轻量化说明.md`。

## 交付映射

- 数据构建：`scripts/generate_data.py` 生成统一 JSONL，含场景、轨迹、三维评价、结构化反思、质量分与数据划分。
- Critic：`src/selfevolve_drive/critics.py` 实现 Safety/Rule/Comfort、多类型 Critic 与证据定位。
- Self-Evolution：`scripts/train.py` 实现 Baseline、SFT、失败样本加权 Reflection SFT 及偏好优化 DPO。
- 多轮进化：本地执行三轮 rollout→Critic→Reflection→失败记忆回放，历史写入 `outputs/evolution_history.json`，经验池写入 `outputs/reflection_memory.jsonl`。
- 完整训练接口：`scripts/prepare_vla_training.py` 导出 OpenDriveVLA SFT、Reflection SFT、DPO 数据；`scripts/train_opendrivevla_peft.py` 提供 LoRA/PEFT 训练入口，Windows 可用 `--check-only` 验证。
- 实验评估：`scripts/evaluate.py` 输出碰撞风险、违章、礼让失败、舒适性与策略误差，并标记未见场景测试集。
- Demo：`demo/index.html` + `demo/styles.css` + `demo/app.js` + `scripts/run_demo.py`，通过 Python API 动态生成双轨迹、Critic、Reflection 与自进化历史。
- 基座查验与适配：`scripts/audit_opendrivevla.py` 检查本地资产，`scripts/import_opendrivevla_outputs.py` 将官方 `plan_conv.json` 转为 Demo 可回放缓存。
- 完整报告：主交付为 `report/自动驾驶VLA自进化研究报告.tex` 与同名 PDF；内容源为
  `report/项目报告.md`。安装 MiKTeX 或 TeX Live（含 XeLaTeX）后运行
  `python tools/build_report_latex.py --compile` 可重新生成 LaTeX 与 PDF。参考论文与源码的
  本地情况见 `references/REFERENCE_MANIFEST.md`。
- GitHub 协作、基座依赖、数据集目录和 GPU 环境说明：见 `docs/GitHub协作与环境配置.md`。

## 重要边界

默认实验是可在 CPU 上复现的研究原型，指标是合成场景上的开放环代理指标，不能冒充 nuScenes/CARLA 闭环成绩。真实基座接入时，应将 `Policy.plan()` 替换为 OpenDriveVLA 推理输出，并保持 JSONL schema、Critic、Reflection、训练数据导出和评估接口不变。
