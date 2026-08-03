# Driving Reflection Demo 使用说明

## 1. 启动

Windows 可直接双击项目根目录的 `启动Demo.bat`。也可在 PowerShell 中运行：

```powershell
python scripts/run_demo.py
```

随后访问 `http://127.0.0.1:8000`。Demo 是带 Python API 的非静态网页，不是双击即运行的静态 HTML。现场模式无需 GPU、数据库或在线 API，仅使用 Python 标准库和 NumPy。

## 2. 现场演示流程

1. 选择“红灯路口”“行人横穿”“前车急缓”或“雨夜弯道”。
2. 选择反思后策略：SFT、Reflection SFT 或 Reflection DPO。
3. 点击“运行 Critic → Reflection → 重规划”。
4. 在中央俯视图比较红色 Baseline 与青色反思后轨迹，使用播放、暂停和重置按钮观察 6 秒预测窗口。
5. 右侧依次讲解三维评分、感知—初始规划—Critic—反思—重规划事件链，以及结构化根因、证据、纠正策略和反事实动作。

推荐课堂演示“红灯路口”：Baseline 往往保持较高目标速度，Critic 定位 `red_light_risk`，反思后策略提前减速。再切换“行人横穿”，展示同一机制如何迁移到礼让约束。

## 3. 展示元素含义

- 红色轨迹：有意保留漏检风险的初始 Baseline，用于产生失败样本。
- 青色轨迹：SFT、Reflection SFT 或 Reflection DPO 的重规划结果。
- Safety / Rule / Comfort：规则 Critic 的三个独立维度；Overall 按 0.45 / 0.35 / 0.20 加权。
- 闭环事件：把一次决策拆为可审计的五阶段日志。
- Driving Reflection：从 Baseline 失败生成的结构化训练知识，而不是模型的隐藏思维链。

## 4. 模型真实性声明

当前网页默认运行 `auto` 模式。它先按 `scene_id` 查找 `outputs/opendrivevla/demo_cache.jsonl` 中的真实 OpenDriveVLA GPU 推理结果；没有匹配记录时，明确回退到 `LiteVLA CPU surrogate`。页面顶部和左侧模型卡会显示本次轨迹的真实来源。

checkpoint 已下载不等于当前网页进程已经加载权重。官方推理依赖 Linux、CUDA、PyTorch、DeepSpeed 和定制 MMCV/MMDet3D；普通 Windows 现场环境采用“GPU 离线推理、Demo 缓存回放”以保证稳定性。

面向正式模型实验，OpenDriveVLA-0.5B 是必要基座。必须先从官方模型页取得访问权限并下载 checkpoint，再通过 LoRA 冻结大部分视觉骨干，仅训练投影层、轨迹头和少量语言层，最后使用本项目生成的 Reflection SFT 与 DPO 数据继续训练。该方案不是从零训练，且比 7B/22GB 级模型更适合课程算力。官方模型网址：https://huggingface.co/OpenDriveVLA/OpenDriveVLA-0.5B

## 5. 常见问题

- 页面无法打开：确认命令窗口仍在运行，并检查 8000 端口是否被占用；可改用 `python scripts/run_demo.py --port 8080`。
- 修改参数后没有变化：点击蓝绿色运行按钮重新请求规划。
- 现场无需联网：所有轨迹与评价均本地计算；只有接入在线 LLM Critic 时才需要配置 API。
- 停止服务：回到启动窗口按 `Ctrl+C`。

## 6. 导入真实 OpenDriveVLA 推理结果

先在官方 Linux/GPU 环境产生 `plan_conv.json`，再准备场景清单 JSONL。每行至少包含官方输出对应的 `id`、Demo 使用的 `scene_id`，以及 `ego_speed`、`speed_limit`、`lead_distance`、`lead_speed`、`traffic_light`、`stopline_distance`、`pedestrian_distance`、`road_curvature`、`route_command`、`weather`。

```powershell
python scripts/import_opendrivevla_outputs.py `
  --plan-conv path\to\plan_conv.json `
  --scenario-manifest path\to\scenario_manifest.jsonl
```

导入后重启 Demo。预设场景 ID 分别为 `demo_red_light`、`demo_pedestrian`、`demo_lead_vehicle` 和 `demo_rainy_curve`；缓存含同名记录时会自动使用真实轨迹。
