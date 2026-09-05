# DroneVehicle RGB-IR 目标检测基准仓库

面向无人机航拍可见光-红外（RGB-IR）双模态车辆检测的多方法对比仓库，基于 [DroneVehicle](https://arxiv.org/abs/2003.02437) 数据集（28,439 对像素级对齐图像，5 类车辆）。

## 仓库结构

```
baselines/
└── yolov11_rgbt/      # YOLOv11-RGBT（Ultralytics 8.3.75 多光谱 fork）+ DroneVehicle RGBT-OBB 实验
methods/
└── dn_dino_dv/        # DN-DETR / DINO（IDEA-Research）适配 DroneVehicle（COCO 格式，HBB 外接框）
datasets_tools/        # DroneVehicle 标注转换 / 分析脚本
```

各子项目相互独立，环境、依赖与训练方式分别见 `baselines/yolov11_rgbt/README.md` 和 `methods/dn_dino_dv/README.md`。

## 主要结果（DroneVehicle 官方测试集，8980 对图像，mAP@IoU=0.5）

| 方法 | 类型 | mAP |
| --- | --- | --- |
| C2Former (TGRS 2024) | RGB-IR，OBB | 72.5 |
| YOLO11n-RGBT-midfusion-obb（`baselines/yolov11_rgbt`） | RGB-IR，OBB | **79.4** |
| DN-DINO（`methods/dn_dino_dv`） | 单模态，HBB | 口径不同，不直接可比；val/HBB 结果见 `methods/dn_dino_dv/results/dv_full_report.md` |

口径说明：DroneVehicle 可见光侧标注在夜景图像上严重缺标，官方评测标签与**红外侧**标注一致，因此所有方法的训练/评测标签均由红外侧标注生成（详见 `baselines/yolov11_rgbt/README.md`）。

## 数据准备流程

原始 DroneVehicle 为 VOC 风格 XML（4 点多边形旋转框）。`datasets_tools/` 提供两条转换链路：

| 脚本 | 用途 |
| --- | --- |
| `transform_DroneVehicle_to_YOLO_OBB.py` | XML → YOLO OBB txt（供 `baselines/yolov11_rgbt` 训练） |
| `xml_to_coco.py` | XML → COCO JSON（OBB 外接水平框，供 `methods/dn_dino_dv` 训练） |
| `prepare_matched.py` | 构建 C2Former 要求的 `*_total/Matched*` 目录（符号链接，不复制图像） |
| `analyze_dv.py` | 检查目录结构与 RGB-IR 配对，输出 OBB 尺寸/角度/类别统计 |

## 声明与引用

- `baselines/yolov11_rgbt` 基于 [YOLOv11-RGBT](https://arxiv.org/abs/2506.14696)（AGPL-3.0，见根目录 `LICENSE`）
- `methods/dn_dino_dv` 基于 [DN-DETR](https://github.com/IDEA-Research/DN-DETR)（Apache-2.0，见其目录内 `LICENSE`）
- 数据集版权归 DroneVehicle 原作者所有；引用格式见 `baselines/yolov11_rgbt/README.md`
