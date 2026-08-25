# AGENTS.md — YOLOv11-RGBT

Guidance for AI coding agents working in this repository. Read this before modifying code.

## Project Overview

**YOLOv11-RGBT** is a research codebase for **multispectral (visible + infrared) object detection**, published as *"YOLOv11-RGBT: Towards a Comprehensive Single-Stage Multispectral Object Detection Framework"* (arXiv:2506.14696). It is a fork of the Ultralytics YOLO codebase (internal version `8.3.75`, `ultralytics/__init__.py`) extended to support multi-channel input beyond standard 3-channel RGB:

- **Gray / Gray16bit** — single-channel 8-bit or 16-bit grayscale (`channels=1`)
- **SimOTM / SimOTMBBS / SimOTMSSS / Gray2BGR** — grayscale converted to 3 channels (`channels=3`)
- **BGR / RGB** — standard 3-channel 8-bit color (`channels=3`)
- **RGBT** — 4-channel RGB + thermal/ir (`channels=4`), with early/mid/late/score fusion and weight-sharing variants
- **RGBRGB6C** — 6-channel paired RGB (`channels=6`)
- **Multispectral / Multispectral_16bit** — arbitrary-channel 8-bit or 16-bit multispectral TIFF (`channels=n`)

The package itself is still the standard Ultralytics library (`pyproject.toml` builds `ultralytics`); the RGBT extensions are layered on top.

License: **AGPL-3.0** (see `LICENSE`). Paper PDF: `YOLOv11-RGBT-2506.14696_v2.pdf`. `README.md` has been rewritten to document the DroneVehicle RGBT-OBB experiments (dataset adaptation, training, benchmark vs C2Former).

## Technology Stack

- **Python ≥ 3.8** (README example uses conda env `pt121_py38`, Python 3.8, PyTorch 1.12.1 + CUDA 11.3; PyTorch ≥ 2.2 also documented)
- **PyTorch** + **torchvision** (installed separately via conda, not in `requirements.txt`)
- Key deps: numpy, opencv-python, matplotlib, pillow, pyyaml, scipy, pandas, seaborn, tqdm, `ultralytics-thop` (FLOPs), plus attention-related: `einops`, `timm`, `efficientnet-pytorch`
- Build system: `setuptools` via `pyproject.toml`; `requirements.txt` is a lighter, fork-specific variant
- Optional extras in `pyproject.toml`: `dev` (pytest, mkdocs), `export` (onnx, openvino, tensorflow...), `solutions`, `logging`, `extra`
- `heatmap_RGBT.py` additionally needs `pytorch_grad_cam`

## Build / Install / Run

```bash
pip install -r requirements.txt   # base dependencies (install PyTorch separately via conda)
pip install -e .                  # editable install; required for the `yolo` CLI and multi-GPU training
```

There is **no compilation step** — this is a pure Python package.

### Entry points

The repo is used through **top-level example scripts** (not a single CLI), each wiring `ultralytics.YOLO` with the right `use_simotm` / `channels` arguments:

| Script | Purpose |
| --- | --- |
| `train.py` | Standard 3-channel training (`use_simotm="RGB", channels=3`) |
| `train_Gray.py` | Grayscale training (`Gray` / `Gray16bit`, `channels=1`) |
| `train_RGBT.py` | RGB+T 4-channel training (`RGBT`, `channels=4`) |
| `train_RGBRGB.py` | 6-channel training (`RGBRGB6C`, `channels=6`) |
| `train_multispectral.py` | Arbitrary-channel multispectral (`Multispectral(_16bit)`) |
| `train_MCF_demo.py` | MCF (Modality Complementary Fusion) strategy demo from the paper |
| `train-cls.py`, `train-rtdetr.py` | Classification / RT-DETR variants |
| `train_RGBT_mine_print.py` | Batch-prints model params / FLOPs |
| `detect*.py` | Inference examples for 1C / 4C / 6C / multispectral / standard modes |
| `val.py`, `val_PGI.py` | Validation (standard / PGI strategy) |
| `export.py` | Model export (ONNX etc.) |
| `get_FPS.py` | FPS benchmark |
| `heatmap_RGBT.py` | Grad-CAM heatmap visualization (uses `pytorch_grad_cam`) |
| `transform_COCO_to_RGBT.py`, `transform_MCF.py`, `transform_PGI.py` | Weight-conversion utilities (e.g. adapt COCO-pretrained 3-channel weights to RGBT models) |

Training example:

```bash
python train_RGBT.py   # edit script: model YAML, data YAML, epochs, device, use_simotm, channels
```

The `yolo` / `ultralytics` CLI entry points (`ultralytics.cfg:entrypoint`) also exist after `pip install -e .`.

## Repository Layout

```
ultralytics/            # the forked Ultralytics package (all real library code)
├── cfg/
│   ├── default.yaml    # all run-time args; RGBT additions: channels, use_simotm, pairs_rgb_ir
│   ├── datasets/       # dataset YAMLs incl. *-rgbt.yaml / *-vis.yaml / *-inf.yaml (FLIR, KAIST, LLVIP, M3FD, VEDAI...)
│   └── models/         # per-architecture dirs; each X has a sibling X-RGBT with multispectral variants
│                     #   v3..v13(+RGBT), 11/11-RGBT, rt-detr, PicoDet, ppyoloe, hyper-yolo, yolox
├── data/               # dataloading — most RGBT modifications live here
│   ├── base.py         # BaseDataset.load_and_preprocess_image() + use_simotm/pairs_rgb_ir handling
│   ├── dataset.py      # classification dataset equivalent
│   ├── loaders.py      # inference-time loaders (LoadImagesAndVideos) with multimodal support
│   ├── build.py        # build_yolo_dataset(..., use_simotm=..., pairs_rgb_ir=...)
│   └── augment.py      # dtype switch for 16-bit modes
├── engine/             # trainer.py, validator.py, predictor.py, model.py, exporter.py, results.py
├── models/             # yolo/, rtdetr/, sam/, fastsam/, nas/ task implementations (train/val/predict per task)
├── nn/
│   ├── tasks.py        # model parsing; DetectionModel(cfg, ch, nc) reads `ch:` from model YAML
│   └── modules/        # conv/block/head/transformer/attention + fork additions (ppyolo, rep_block, yolov13_block)
├── utils/              # loss, metrics, plotting (multimodal annotation), ops, tal, callbacks/
├── solutions/, trackers/, hub/, assets/
tests/                  # upstream pytest suite (test_engine, test_python, test_cuda, test_exports, ...)
examples/               # community inference examples (C++/Rust/ONNXRuntime...), not core
docs/                   # MkDocs documentation sources (docs.ultralytics.com content)
docker/                 # Dockerfiles for many targets (cpu, conda, jetson, jupyter, runner...)
.github/workflows/      # ci.yml, format.yml, docs.yml, docker.yml, publish.yml, ...
```

## Key Architecture Facts (read before editing)

1. **Two extra run-time args** (in `ultralytics/cfg/default.yaml:131-134`) drive everything:
   - `channels: int` — model input channels; **must match `ch:` in the model YAML** (e.g. `ch: 4` in `ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion.yaml`).
   - `use_simotm: str` — preprocessing/input mode: `Gray`, `Gray16bit`, `SimOTM`, `SimOTMBBS`, `SimOTMSSS`, `Gray2BGR`, `BGR`/`RGB`, `RGBT`, `RGBRGB6C`, `Multispectral`, `Multispectral_16bit`.
   - `pairs_rgb_ir: ['visible', 'infrared']` — directory-name pair used to derive the IR path from the RGB path (string replacement of element 1 → element 2 in the image path).
2. **The core data-pairing mechanism**: `load_and_preprocess_image()` in `ultralytics/data/base.py` (and equivalents in `dataset.py`, `loaders.py`) loads the visible image, replaces `visible` with `infrared` in the path (via `pairs_rgb_ir`), loads the IR image, and concatenates channels. **This means visible and infrared directories must be sibling folders with identical filenames.** Changing this mechanism affects training, validation, and inference simultaneously.
3. The `use_simotm` / `pairs_rgb_ir` args are threaded explicitly through `engine/predictor.py`, `models/yolo/{detect,segment,pose,classify}/{train,val}.py`, and `models/rtdetr/{train,val}.py` — when adding a new mode, update **all** of these call sites plus the mode dispatch in `base.py`/`dataset.py`/`loaders.py`.
4. 16-bit modes (`Gray16bit`, `Multispectral_16bit`) use `float32` tensors instead of `uint8` (see `ultralytics/data/augment.py`), which affects caching and augmentation dtype handling.
5. Model YAML files in `ultralytics/cfg/models/*-RGBT/` encode the fusion strategy (early/mid/late/score fusion) directly in the architecture — new fusion modules must be registered in `ultralytics/nn/tasks.py`'s `parse_model()` and exported in `ultralytics/nn/modules/__init__.py`.

## Dataset Conventions

- Visible and infrared folders must be **peers** (same parent), with identical relative file names. Four supported layouts (FLIR-style, KAIST-style, images/labels split, and TXT-file lists) are documented in `README.md`; dataset YAMLs live in `ultralytics/cfg/datasets/` (`*-rgbt.yaml` point at the visible path; IR is derived automatically).
- TXT-list configs must contain the `visible` path component so it can be replaced.
- Multispectral modes read multi-channel TIFF files directly.

## Testing

- Framework: **pytest**, configured in `pyproject.toml` (`[tool.pytest.ini_options]`, `--doctest-modules --durations=30`).
- Custom marker: `slow` — slow tests are skipped unless `--slow` is passed (`tests/conftest.py`).
- Run: `pytest tests/` (full suite needs GPU/datasets for some tests; CI runs on schedule via `.github/workflows/ci.yml`).
- Note: the test suite is inherited from upstream Ultralytics; the RGBT top-level scripts are **not** covered by it. Practical verification for RGBT changes is running a short training/val pass, e.g. `python val.py` to check dataloading, or a few epochs of `train_RGBT.py` on a small dataset (`LLVIP-rgbt.yaml`, `KAIST8-rgbt.yaml`).
- Coverage config: `[tool.coverage.run]` in `pyproject.toml` (source `ultralytics/`, data file `tests/.coverage`).

## Code Style

- Line length **120** everywhere: `yapf` (pep8 base, see `[tool.yapf]`), `ruff` (`line-length = 120`, `docstring-code-format`), `isort` (`line_length = 120`), `docformatter` (wrap at 120).
- **Google-style docstrings** (required by `CONTRIBUTING.md`); CI enforces formatting via `.github/workflows/format.yml` (ruff + docformatter + codespell).
- Spell check: `codespell` config in `pyproject.toml` with project ignore-list.
- Source files carry the header `# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license`.
- Comments and docs are predominantly **English**; the fork's added scripts and README sections mix in some **Chinese** comments (both are acceptable in this repo; match the surrounding file).
- Docstring style convention in this codebase: `Args:` / `Returns:` blocks with types in parentheses.

## CI / Deployment

- `.github/workflows/ci.yml` — matrix CI (upstream Ultralytics config: HUB, Tests, GPU, Raspberry Pi, Conda jobs). Some jobs are gated on `github.repository == 'ultralytics/ultralytics'`, so they no-op on forks.
- `format.yml` — auto-formatting on PRs; `docs.yml` builds MkDocs (`mkdocs.yml`); `docker.yml` builds images from `docker/`; `publish.yml` publishes the `ultralytics` package to PyPI (upstream-gated).
- Contribution flow (fork-and-PR, CLA signing) is described in `CONTRIBUTING.md`.

## Security & Operational Notes

- **No git repository** is initialized in this working copy; do not assume VCS operations are available.
- Never commit datasets, model weights (`*.pt`), or run outputs (`runs/`) — they are large and expected to live outside the repo.
- `pip install -e .` installs a `yolo` console script into the environment — be aware when multiple ultralytics checkouts exist on one machine.
- Default hyperparameters, logging integrations (tensorboard, comet, W&B via callbacks), and HUB features are inherited from Ultralytics; HUB/upload features should not be triggered unintentionally.
- Many example scripts contain hard-coded absolute dataset paths (e.g. `E:/BaiduNetdiskDownload/...`, `G:\wan\data\...`) — treat them as examples; users must edit paths for their environment.
