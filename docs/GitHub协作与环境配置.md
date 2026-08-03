# GitHub 协作与环境配置

## 0. Windows 与 Linux 依赖边界

Windows 本地 Demo、模型文件审计和训练数据准备只安装：

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-model.txt
```

不要在 Windows 上执行 `pip install -r requirements-gpu.txt` 来尝试搭建完整
OpenDriveVLA 训练环境。官方定制 MMCV/MMDet3D 需要 Linux、CUDA 和编译工具链。

`requirements-gpu.txt` 是本项目 OpenDriveVLA LoRA/PEFT 的兼容版本清单，其版本已与
OpenDriveVLA 官方 `pyproject.toml` 对齐。DeepSpeed 已移至独立的
`requirements-deepspeed.txt`，因为它的构建必须发生在 Torch 安装之后。官方推理入口
也会导入 DeepSpeed，因此实时推理与训练都应在 Linux/CUDA 中分阶段安装：

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
DS_BUILD_OPS=0 pip install --no-build-isolation -r requirements-deepspeed.txt
pip install -r requirements-gpu.txt
```

如果此前出现 `Unable to pre-compile ops without torch installed`，原因是 pip 的隔离
构建环境在解析 DeepSpeed 时看不到尚未安装的 Torch；不能依靠同一个 requirements
文件中的排列顺序解决。

## 1. 仓库内容边界

GitHub 主分支保存项目源码、配置、测试、使用文档、轻量模型参数、指标摘要、报告，以及
LiteVLA Demo 直接运行所需的 `data/reflection_dataset.jsonl`（5,112 条 nuScenes
衍生记录）和 `data/nuscenes_manifest.json`。以下内容不提交：

- OpenDriveVLA-0.5B 权重和 tokenizer 大文件；
- OpenDriveVLA 官方源码的本地副本与定制 OpenMMLab 依赖；
- nuScenes、CAN bus、地图、UniAD info 和评估 GT；
- OpenDriveVLA SFT/Reflection SFT/DPO 训练导出 JSONL 和反思记忆；
- Hugging Face token、API key、`.env` 与本地 QA 文件。

## 2. 克隆后最快运行方式

仅运行 Windows 轻量 Demo：

```powershell
python -m pip install -r requirements.txt
python scripts/run_demo.py
```

访问 `http://127.0.0.1:8000`。主分支已包含完整 5,112 条 LiteVLA Demo JSONL，
不需要先执行数据生成。前端由 `demo/index.html`、CSS 文件和 `demo/app.js` 组成，
本身不保存固定场景、规划轨迹或评分。场景来自完整训练数据索引，规划与评分由 Python API 实时返回。

主要接口：

- `GET /api/data/status`：完整 JSONL 索引状态、记录数与数据划分；
- `GET /api/scenarios/presets`：从训练数据动态匹配四类演示场景；
- `GET /api/scenarios?offset=0&limit=50`：分页访问训练场景；
- `GET /api/scenario?id=scene-000000`：加载指定场景；
- `GET /api/logs?after=0`：增量读取数据、模型、Critic 和 Reflection 日志；
- `POST /api/compare`：Baseline 与反思策略实时对比；
- `POST /api/model/plan`：统一规划入口，请求体与 `/api/evaluate` 相同；
- `GET /api/model/status`：checkpoint、代码、依赖和缓存状态。

`/api/model/plan` 的 `runtime` 可取 `auto`、`cache` 或 `lite`。其中 `cache` 是严格模式：找不到由真实 OpenDriveVLA 推理导入的匹配轨迹时返回错误，不会静默生成 LiteVLA 结果；只有 `auto` 允许在无缓存时退回 CPU 轻量基座。

## 3. 下载冻结基座

本项目指定 OpenDriveVLA-0.5B，不从零训练。模型页面：

https://huggingface.co/OpenDriveVLA/OpenDriveVLA-0.5B

授权后运行：

```powershell
python -m pip install -r requirements-model.txt
python scripts/download_base_model.py
python scripts/audit_opendrivevla.py
```

权重目标目录为 `references/models/OpenDriveVLA-0.5B/`。审计结果应满足：

- `checkpoint_installed=true`；
- `safetensors_header_ok=true`；
- `architecture=LlavaQwenForCausalLM`；
- `code_complete=true`（下载官方代码后）。

下载权重只表示基座资产完整，不等于当前进程已执行视觉推理。调用 `GET /api/model/status` 可区分 checkpoint、官方代码、GPU运行环境和真实缓存状态。

## 4. 官方代码与 GPU 依赖

将官方仓库克隆到以下任一路径，项目会自动定位：

```text
references/repositories/OpenDriveVLA/
references/repos/OpenDriveVLA/
references/models/VLA-code/OpenDriveVLA-code/
```

推荐路径：

```bash
git clone https://github.com/DriveVLA/OpenDriveVLA.git references/repositories/OpenDriveVLA
```

完整推理/PEFT 推荐 Linux、Python 3.10、NVIDIA CUDA 和 PyTorch 2.1.2。官方项目还要求从其 `third_party/` 编译 MMCV 1.7.2 与 mmdetection3d 1.0.0rc6，并安装 DeepSpeed、Transformers、MMDet、MMSeg 等依赖。请优先遵循官方 `docs/1_INSTALL.md`，不要在 Windows Demo 环境中强行安装定制 CUDA 包。

完整 GPU 训练入口保留如下；它需要研究者自行准备与官方格式兼容的阶段数据：

```powershell
python scripts/train_opendrivevla_peft.py --stage sft --check-only
python scripts/train_opendrivevla_peft.py --stage reflection_sft --check-only
python scripts/train_opendrivevla_peft.py --stage dpo --check-only
```

Linux/CUDA 环境确认无误后，去掉 `--check-only` 执行对应阶段。

## 5. nuScenes 数据需求

本课程轻量实验使用 nuScenes v1.0-mini。远端已经包含抽取后的 5,112 条
`data/reflection_dataset.jsonl`，协作者无需下载或生成数据即可运行：

```powershell
python scripts/train.py
python scripts/evaluate.py
python scripts/run_demo.py
```

已交付记录保留 `sample_token`、六相机 `sample_data_token`/图像引用、
`annotation_token`、scene/log/location 和 6 秒未来轨迹来源。一次性下载、抽取及数据生成代码
不随主分支分发；原始数据也不会提交，缺少原图时相机预览会自动隐藏。

完整 OpenDriveVLA 官方推理与评估需要：

- nuScenes v1.0 trainval/full：`samples/`、`sweeps/`、`v1.0-trainval/`；
- CAN bus expansion；
- Map expansion v1.3；
- UniAD temporal train/val info；
- `cached_nuscenes_info.pkl`；
- 评估 GT：轨迹、mask、规划分割和 VAD 分割缓存。

建议目录：

```text
references/repositories/OpenDriveVLA/data/
├─ infos/
│  ├─ nuscenes_infos_temporal_train.pkl
│  └─ nuscenes_infos_temporal_val.pkl
├─ nuscenes/
│  ├─ can_bus/
│  ├─ maps/
│  ├─ samples/
│  ├─ sweeps/
│  ├─ v1.0-trainval/
│  └─ cached_nuscenes_info.pkl
└─ eval_share/gt/
   ├─ gt_traj.pkl
   ├─ gt_traj_mask.pkl
   ├─ planing_gt_segmentation_val/
   └─ vad_gt_seg.pkl
```

数据集受其各自许可约束，不得提交到本 GitHub 仓库。

## 6. 三种 Demo 模式

- `auto`：优先使用匹配的 OpenDriveVLA 真实推理缓存，无匹配时明确回退到 LiteVLA。
- `cache`：严格要求真实缓存；缺失时返回错误，不静默回退。
- `lite`：始终使用 Windows CPU 轻量自进化策略。

GPU 推理结果通过以下命令导入：

```powershell
python scripts/import_opendrivevla_outputs.py `
  --plan-conv path/to/plan_conv.json `
  --scenario-manifest path/to/scenario_manifest.jsonl
```

默认缓存路径为 `outputs/opendrivevla/demo_cache.jsonl`，也可通过 `OPENDRIVEVLA_CACHE` 环境变量覆盖。

## 7. 协作规范

- 不提交 token、基座权重、真实数据集或未审核的训练导出；LiteVLA Demo 的
  `data/reflection_dataset.jsonl` 是明确纳入主分支的运行资产；
- 修改 schema 或 API 时同步更新测试和 `docs/Demo使用说明.md`；
- 提交前运行 `python -m unittest discover -s tests -v`；
- 运行 `python scripts/audit_opendrivevla.py` 核对本地模型状态；
- 真实 GPU 与轻量代理指标必须在输出和报告中明确区分。
