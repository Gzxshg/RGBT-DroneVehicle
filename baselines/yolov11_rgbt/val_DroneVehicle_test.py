# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# DroneVehicle RGBT OBB：在 test 集（8980 对）上评估训练好的 best.pt
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/DroneVehicle/DroneVehicle-yolo11n-RGBT-midfusion-obb-2/weights/best.pt')  # 红外侧全量标签训练版
    model.val(data='ultralytics/cfg/datasets/DroneVehicle-rgbt-obb.yaml',
              split='test',  # 使用 test split
              imgsz=640,
              batch=16,
              workers=8,
              device='0',
              use_simotm="RGBT",  # 4通道 RGB + IR
              channels=4,
              pairs_rgb_ir=['img', 'imgr'],  # testimg->testimgr
              project='runs/DroneVehicle',
              name='test-eval-',
              )
