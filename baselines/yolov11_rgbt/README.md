# 基于 YOLOv11-RGBT 的 DroneVehicle 可见光-红外双模态旋转目标检测

本仓库将 RGB-T 双流融合检测模型 **YOLO11n-RGBT（mid-fusion + OBB 旋转框检测头）** 应用于无人机航拍可见光-红外数据集 **DroneVehicle**，并在官方评测口径下与 **C2Former** 进行了对比实验。

代码框架基于 [YOLOv11-RGBT](https://arxiv.org/abs/2506.14696)（Ultralytics 8.3.75 的多光谱扩展版），在此之上完成了 DroneVehicle 数据集的适配、标注口径修正、训练与基准测试。

## 主要结果（DroneVehicle 官方测试集，8980 对图像，159616 个标注目标）

| 方法 | car | truck | bus | van | freight_car | **mAP（IoU=0.5）** |
| --- | --- | --- | --- | --- | --- | --- |
| C2Former (TGRS 2024) | 90.3 | 71.6 | 89.3 | 53.7 | 57.6 | 72.5 |
| **本仓库（YOLO11n-RGBT-midfusion-obb）** | **98.5** | **78.6** | **95.2** | **60.0** | **64.7** | **79.4** |

总体 mAP 提升 **+6.9 个百分点**，5 个类别全部领先。本模型参数量 3.87 M，FLOPs 9.75 G（imgsz=640），单张 RTX 3090 推理约 2.6 ms/帧。

## 环境部署

本实验在以下环境完成验证（其他版本未逐一测试）：

- 操作系统：Linux；GPU：RTX 3090 ×4（单卡亦可训练）
- Python 3.8，PyTorch 1.9.0+cu111，torchvision 0.10.0+cu111

```bash
conda create -n rgbt python=3.8 -y
conda activate rgbt
# 安装 PyTorch（按本机 CUDA 版本选择，RTX 3090 需 cu111 及以上）
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# 安装依赖与本仓库
pip install -r requirements.txt
pip install pandas seaborn tqdm psutil py-cpuinfo "ultralytics-thop>=2.0.0" requests albumentations
pip install -e .
```

**版本兼容注意事项**：

- `ultralytics/nn/tasks.py` 在导入时硬依赖 `timm.models.focalnet`，而新版 timm（≥1.0）使用了 `torch.fx.wrap`，在 torch 1.9 下无法导入。本环境固定 **`timm==0.9.12`** 以兼容 torch 1.9；若使用 torch ≥ 1.12，新版 timm 无此限制。
- torch 1.9 下 `torch.meshgrid(indexing=...)` 不支持，本仓库检测主路径（Conv/C3k2/C2PSA/OBB）已做版本守卫，不受影响；RT-DETR 的 AIFI 模块在 torch 1.9 下不可用。

## 数据集准备

### 数据集结构

使用 [DroneVehicle](https://arxiv.org/abs/2003.02437) 数据集（28,439 对像素级对齐的可见光-红外图像，5 类车辆目标）。期望的目录结构：

```
DroneVehicle/
├── train/  trainimg/  trainimgr/  trainlabel/  trainlabelr/     # 17990 对
├── val/    valimg/    valimgr/    vallabel/    vallabelr/       # 1469 对
└── test/   testimg/   testimgr/   testlabel/   testlabelr/      # 8980 对
```

其中 `*img` 为可见光图像，`*imgr` 为红外图像；`*label` / `*labelr` 分别为可见光侧与红外侧的 XML 标注（4 点多边形旋转框，部分目标为水平框 bndbox）。

### 标注口径说明（重要）

经逐文件核对：**DroneVehicle 官方评测标签（`*_total/*MatchedLabelTxtMVP_total`）与红外侧 XML 标注完全一致**（目标数与坐标逐点吻合）。可见光侧 XML 在夜景图像上存在严重缺标——可见光画面几乎全黑时标注为 0 个目标，而红外侧同一图像可标注出数十个目标（例如测试集 `06335`：可见光侧 0 个 vs 红外侧 77 个）。因此本仓库的训练与评测均基于**红外侧标注**生成标签，与官方口径一致（测试集 GT 总数 159616，与官方相同）。

### 标注转换

原始 XML 标注需转换为 YOLO OBB 格式（`class x1 y1 x2 y2 x3 y3 x4 y4`，归一化）。转换脚本已内置该口径：

```bash
python transform_DroneVehicle_to_YOLO_OBB.py
```

- 标签以 `.txt` 形式直接写入可见光图像目录（与 `.jpg` 同名），红外侧无需标签（配对关系由训练参数 `pairs_rgb_ir=['img','imgr']` 自动推导）
- 仅有水平框（bndbox）的目标会被转换为等价的 4 点多边形
- 原始数据中 `feright_car` / `feright car` 两种拼写统一归并为 `freight_car`
- 转换统计：train 17990 对 / 316410 目标，val 1469 对 / 24490 目标，test 8980 对 / 159616 目标；28439 张图中共 40 个无效目标（无框或类别名乱标）被跳过

数据集配置文件：`ultralytics/cfg/datasets/DroneVehicle-rgbt-obb.yaml`。

## 训练

```bash
python train_DroneVehicle_RGBT_OBB.py
```

训练配置（`train_DroneVehicle_RGBT_OBB.py`）：

| 配置项 | 值 |
| --- | --- |
| 模型 | `ultralytics/cfg/models/11-RGBT/yolo11n-RGBT-midfusion-obb.yaml`（可见光/红外双流 backbone，P3/P4/P5 特征通道拼接融合，OBB 检测头，`ch: 4`） |
| 输入模式 | `use_simotm="RGBT"`，`channels=4`（RGB 3 通道 + 红外 1 通道） |
| 图像尺寸 | 640 × 640 |
| 优化器 | SGD（lr0=0.01, momentum=0.937, weight_decay=0.0005），cosine 学习率衰减 |
| batch | 64（4 卡 DDP，每卡 16；Ultralytics DDP 中 `batch` 为总 batch，多卡会自动均分，单卡训练请改回 16） |
| epochs | 100（`close_mosaic=10`，最后 10 轮关闭 mosaic） |
| 硬件 / 耗时 | 4 × RTX 3090，约 1.97 小时 |

训练日志与曲线保存在 `runs/DroneVehicle/DroneVehicle-yolo11n-RGBT-midfusion-obb-*/`，最佳权重为 `weights/best.pt`。

## 测试

```bash
python val_DroneVehicle_test.py   # split='test'，使用官方口径测试集
```

最终结果（`runs/DroneVehicle/test-eval-*/`）：

| 指标 | 验证集 val（1469 对） | 测试集 test（8980 对） |
| --- | --- | --- |
| Precision | 80.2 | 76.1 |
| Recall | 76.2 | 77.7 |
| mAP50 | 81.6 | **79.4** |
| mAP50-95 | 64.4 | 62.8 |

测试集各类别 mAP50 / mAP50-95：car 98.5 / 81.6，truck 78.6 / 58.6，bus 95.2 / 77.0，van 60.0 / 47.7，freight_car 64.7 / 49.0。

## 与 C2Former 对比的口径说明

- C2Former 结果取自其 mmrotate 实现在同一 DroneVehicle 官方测试集上的评测日志（`eval_epoch24.log`，mAP=72.5），两边 GT 逐类一致（各类目标数差异 ≤5，源自个别重复框的去除）
- C2Former 侧实际参与评测的图像为 8962 张（比完整测试集少 18 张，影响可忽略）
- mmrotate 的 `mAP` 与 YOLO 的 `mAP50` 均为 IoU=0.5 阈值下的 AP 均值，插值细节略有差异，属可比的同口径指标

## 本仓库新增/修改的文件

| 文件 | 说明 |
| --- | --- |
| `transform_DroneVehicle_to_YOLO_OBB.py` | DroneVehicle XML 标注 → YOLO OBB 转换脚本（基于红外侧标注） |
| `train_DroneVehicle_RGBT_OBB.py` | 训练脚本（4 通道 RGBT + midfusion + OBB） |
| `val_DroneVehicle_test.py` | 测试集评估脚本 |
| `ultralytics/cfg/datasets/DroneVehicle-rgbt-obb.yaml` | 数据集配置 |

其余为 YOLOv11-RGBT 原框架代码。模型结构（双流 midfusion 等）的说明见原框架论文与 `AGENTS.md`。

## 声明与引用

本仓库基于 [YOLOv11-RGBT](https://arxiv.org/abs/2506.14696)（其本身为 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 的多光谱 fork）构建，遵循 **AGPL-3.0** 许可证（见 `LICENSE`）。数据集版权归 DroneVehicle 原作者所有。若本仓库对您有帮助，请引用以下工作：

```bibtex
@article{yolov11rgbt,
  title={YOLOv11-RGBT: Towards a Comprehensive Single-Stage Multispectral Object Detection Framework},
  journal={arXiv preprint arXiv:2506.14696},
  year={2025}
}

@article{dronevehicle,
  title={Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning},
  author={Sun, Yiming and Cao, Bing and Zhu, Pengfei and Hu, Qinghua},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2022}
}

@article{c2former,
  title={C2Former: Calibrated and Complementary Transformer for RGB-Infrared Object Detection},
  author={Yuan, Maoxun and Wei, Xingxing},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024}
}
```
