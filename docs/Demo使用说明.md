# Driving Reflection Demo 使用说明

## 1. 启动

Windows 可直接双击项目根目录的 `启动Demo.bat`。也可在 PowerShell 中运行：

```powershell
python scripts/run_demo.py
```

随后访问 `http://127.0.0.1:8000`。Demo 是带 Python API 的非静态网页，不是双击即运行的静态 HTML。现场模式无需 GPU、数据库或在线 API，仅使用 Python 标准库和 NumPy。

## 2. 现场演示流程

1. 选择由完整训练数据索引动态匹配出的“红灯路口”“行人横穿”“近距前车”或“恶劣弯道”，也可输入任意 `scene_id` 或随机抽取样本。
2. 选择反思后策略：SFT、Reflection SFT 或 Reflection DPO。
3. 点击“运行 Critic → Reflection → 重规划”。
4. 在中央俯视图比较红色 Baseline 与青色反思后轨迹，使用播放、暂停和重置按钮观察 6 秒预测窗口。
5. 右侧依次讲解三维评分、评分数据近邻、后端实时日志、感知—初始规划—Critic—反思—重规划事件链，以及结构化根因、证据、纠正策略和反事实动作。

推荐课堂演示“红灯路口”：Baseline 往往保持较高目标速度，Critic 定位 `red_light_risk`，反思后策略提前减速。再切换“行人横穿”，展示同一机制如何迁移到礼让约束。

## 3. 展示元素含义

- 红色轨迹：有意保留漏检风险的初始 Baseline，用于产生失败样本。
- 青色轨迹：SFT、Reflection SFT 或 Reflection DPO 的重规划结果。
- Safety / Rule / Comfort：每次请求都用当前轨迹实时计算。单维评分由规则 Critic 65%、训练后的 Reward Critic 25% 和完整训练数据的七近邻校准 10% 组成；Overall 再按 0.45 / 0.35 / 0.20 汇总。
- 后端实时日志：显示完整数据文件加载、Reward Critic 参数连接、规划模型调用、实际 runtime、耗时、评分近邻和 Reflection 结果。
- 闭环事件：把一次决策拆为可审计的五阶段可视化摘要。
- Driving Reflection：从 Baseline 失败生成的结构化训练知识，而不是模型的隐藏思维链。

## 4. 模型真实性声明

当前网页默认运行 `auto` 模式。它先按 `scene_id` 查找 `outputs/opendrivevla/demo_cache.jsonl` 中的真实 OpenDriveVLA GPU 推理结果；没有匹配记录时，明确回退到 `LiteVLA CPU surrogate`。页面顶部和左侧模型卡会显示本次轨迹的真实来源。

checkpoint 已下载不等于当前网页进程已经加载权重。官方推理入口会导入 DeepSpeed，并依赖 Linux、CUDA、PyTorch 和定制 MMCV/MMDet3D。普通 Windows 现场环境采用“GPU 离线推理、Demo 缓存回放”以保证稳定性。

面向正式模型实验，OpenDriveVLA-0.5B 是必要基座。必须先从官方模型页取得访问权限并下载 checkpoint，再通过 LoRA 冻结大部分视觉骨干，仅训练投影层、轨迹头和少量语言层，最后使用本项目的轻量化代理训练数据执行 Reflection SFT 与 DPO。该方案不是从零训练，且比 7B/22GB 级模型更适合课程算力。官方模型网址：https://huggingface.co/OpenDriveVLA/OpenDriveVLA-0.5B

## 5. OpenDriveVLA-0.5B 的 CPU 边界

0.5B 语言骨干的权重体积本身允许在大内存 CPU 主机上加载，但官方 OpenDriveVLA 不是单纯的 0.5B 文本模型。其 `mm_vision_tower=uniad_track_map`，官方推理入口固定选择 CUDA、使用 CUDA autocast、DeepSpeed、NCCL/分布式路径，并依赖带 CUDA 算子的定制 MMCV/MMDet3D；UniAD 插件中也存在直接 `.cuda()` 的实现。因此当前官方端到端视觉规划代码不能在未修改的 Windows CPU 环境可靠运行。

若运行机器具备兼容的 NVIDIA GPU，实时运行完整基座推荐使用 WSL2/Ubuntu 或独立 Linux 环境，安装官方版本的 PyTorch 2.1.2、DeepSpeed 0.14.2，并编译仓库 `third_party/` 中的 MMCV/MMDet3D。Windows Demo 保持使用训练后的 LiteVLA 策略或导入真实 GPU 推理缓存，两者都会在页面明确标记，不能将 CPU 代理结果标成 OpenDriveVLA 权重实时推理。

## 6. 常见问题

- 页面无法打开：确认命令窗口仍在运行，并检查 8000 端口是否被占用；可改用 `python scripts/run_demo.py --port 8080`。
- 修改参数后没有变化：点击蓝绿色运行按钮重新请求规划。
- 现场无需联网：所有轨迹与评价均本地计算；只有接入在线 LLM Critic 时才需要配置 API。
- 停止服务：回到启动窗口按 `Ctrl+C`。

## 7. 导入真实 OpenDriveVLA 推理结果

先在官方 Linux/GPU 环境产生 `plan_conv.json`，再准备场景清单 JSONL。每行至少包含官方输出对应的 `id`、Demo 使用的 `scene_id`，以及 `ego_speed`、`speed_limit`、`lead_distance`、`lead_speed`、`traffic_light`、`stopline_distance`、`pedestrian_distance`、`road_curvature`、`route_command`、`weather`。

```powershell
python scripts/import_opendrivevla_outputs.py `
  --plan-conv path\to\plan_conv.json `
  --scenario-manifest path\to\scenario_manifest.jsonl
```

导入后重启 Demo。场景按钮不是固定 ID，而是从 `data/reflection_dataset.jsonl` 动态选取；真实缓存必须使用与当前训练样本相同的 `scene_id` 和策略名才能命中。用户修改任一场景参数后，前端会使用带 `:edited` 后缀的新 ID，防止误回放不匹配的缓存轨迹。
