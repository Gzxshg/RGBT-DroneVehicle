# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# DroneVehicle RGBT 旋转框(OBB)训练示例：可见光+红外 4 通道 mid-fusion + OBB 检测头
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11-RGBT/yolo11n-RGBT-midfusion-obb.yaml')
    # model.load('yolo11n.pt')  # 如需加载 COCO 预训练权重可先用 transform_COCO_to_RGBT.py 转换
    model.train(data='ultralytics/cfg/datasets/DroneVehicle-rgbt-obb.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=64,  # 4 卡 DDP：总 batch=64（每卡 16），过小将无法发挥多卡加速
                close_mosaic=10,
                workers=8,
                device='0,1,2,3',  # 4 卡 DDP；batch 为总 batch，会自动除以卡数
                optimizer='SGD',  # using SGD
                # lr0=0.002,
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.02,  # 调试用：只用部分数据
                pairs_rgb_ir=['img', 'imgr'],  # trainimg->trainimgr / valimg->valimgr / testimg->testimgr
                use_simotm="RGBT",
                channels=4,
                project='runs/DroneVehicle',
                name='DroneVehicle-yolo11n-RGBT-midfusion-obb-',
                )
