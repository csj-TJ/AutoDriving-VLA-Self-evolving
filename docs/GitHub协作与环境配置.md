# GitHub 协作与环境配置

## 1. 仓库内容边界

GitHub 主分支保存项目源码、配置、测试、使用文档、轻量模型参数、指标摘要和报告。以下内容不提交：

- OpenDriveVLA-0.5B 权重和 tokenizer 大文件；
- OpenDriveVLA 官方源码的本地副本与定制 OpenMMLab 依赖；
- nuScenes、CAN bus、地图、UniAD info 和评估 GT；
- 5K JSONL、SFT/Reflection SFT/DPO 全量生成数据和反思记忆；
- Hugging Face token、API key、`.env` 与本地 QA 文件。

## 2. 克隆后最快运行方式

仅运行 Windows 轻量 Demo：

```powershell
python -m pip install -r requirements.txt
python scripts/run_all.py --samples 5000
python scripts/run_demo.py
```

访问 `http://127.0.0.1:8000`。前端由 `demo/index.html`、`demo/styles.css`、`demo/app.js` 组成，通过 `/api/compare`、`/api/meta`、`/api/model/status` 和 `/api/evolution` 调用 Python 后端。协作者也可调用 `POST /api/model/plan` 作为统一规划入口，其请求体与 `/api/evaluate` 相同。

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

训练资产校验：

```powershell
python scripts/prepare_vla_training.py
python scripts/train_opendrivevla_peft.py --stage sft --check-only
python scripts/train_opendrivevla_peft.py --stage reflection_sft --check-only
python scripts/train_opendrivevla_peft.py --stage dpo --check-only
```

Linux/CUDA 环境确认无误后，去掉 `--check-only` 执行对应阶段。

## 5. nuScenes 数据需求

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

- 不提交 token、权重、数据集和生成 JSONL；
- 修改 schema 或 API 时同步更新测试和 `docs/Demo使用说明.md`；
- 提交前运行 `python -m unittest discover -s tests -v`；
- 运行 `python scripts/audit_opendrivevla.py` 核对本地模型状态；
- 真实 GPU 与轻量代理指标必须在输出和报告中明确区分。
