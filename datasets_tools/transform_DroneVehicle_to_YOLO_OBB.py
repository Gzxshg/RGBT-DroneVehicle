# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
DroneVehicle 数据集标注转换脚本：VOC 风格 XML（4 点多边形旋转框）→ YOLO OBB 格式 txt。

原始结构：
    DroneVehicle/{train,val,test}/{trainimg,trainimgr,trainlabel,trainlabelr,...}
输出（YOLO OBB 标签，直接写到可见光图片旁边，与 .jpg 同名 .txt；红外侧不需要标签，
    对应仓库 README 的 Method 1 布局，配对由 pairs_rgb_ir=['img','imgr'] 完成）：
    DroneVehicle/{train,val,test}/{trainimg,valimg,testimg}/*.txt

类别映射（原始数据存在 feright_car / feright car 两种拼写，统一为 freight_car）：
    car: 0, truck: 1, bus: 2, van: 3, freight_car: 4
"""

import xml.etree.ElementTree as ET
from pathlib import Path

# ------------------------------------------------------------------------------------------------------------------
ROOT = Path("/mnt/dataset/DroneVehicle")

# split -> (可见光图片目录, XML 标签目录)；txt 标签直接输出到可见光图片目录内
# 注意：使用红外侧标注（*labelr）——经核对，DroneVehicle 官方 matched 标签与红外侧 XML 完全一致；
# 可见光侧 XML 在夜景图片上严重缺标（几乎全黑时标 0 个目标），不可用作训练/评测基准
SPLITS = {
    "train": (ROOT / "train" / "trainimg", ROOT / "train" / "trainlabelr"),
    "val": (ROOT / "val" / "valimg", ROOT / "val" / "vallabelr"),
    "test": (ROOT / "test" / "testimg", ROOT / "test" / "testlabelr"),
}

CLASS_MAP = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "van": 3,
    "feright_car": 4,  # 原始标注拼写 1
    "feright car": 4,  # 原始标注拼写 2
}
NAMES = ["car", "truck", "bus", "van", "freight_car"]
# ------------------------------------------------------------------------------------------------------------------


def convert_one_xml(xml_path: Path):
    """解析单个 XML，返回 (txt_lines, n_skipped, unknown_names)。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w, h = int(size.findtext("width")), int(size.findtext("height"))
    lines, n_skipped, unknown = [], 0, set()
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        if name not in CLASS_MAP:
            unknown.add(name)
            n_skipped += 1
            continue
        poly = obj.find("polygon")
        if poly is not None:
            # 4 点旋转框，点的顺序保持原始四边形顺序（x1y1 -> x4y4）
            try:
                coords = [float(poly.findtext(f"{axis}{i}")) for i in range(1, 5) for axis in ("x", "y")]
            except (TypeError, ValueError):
                n_skipped += 1
                continue
        else:
            bbox = obj.find("bndbox")
            if bbox is None:
                n_skipped += 1
                continue
            # 部分目标只有水平框 bndbox，转成等价的 4 点多边形（OBB 的特例）
            try:
                xmin, ymin = float(bbox.findtext("xmin")), float(bbox.findtext("ymin"))
                xmax, ymax = float(bbox.findtext("xmax")), float(bbox.findtext("ymax"))
            except (TypeError, ValueError):
                n_skipped += 1
                continue
            coords = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
        # 归一化并裁剪到 [0, 1]
        norm = []
        for j, v in enumerate(coords):
            v = v / w if j % 2 == 0 else v / h
            norm.append(min(max(v, 0.0), 1.0))
        lines.append(f"{CLASS_MAP[name]} " + " ".join(f"{v:.6f}" for v in norm))
    return lines, n_skipped, unknown


def main():
    total_imgs, total_objs, total_skipped = 0, 0, 0
    all_unknown = set()
    for split, (img_dir, xml_dir) in SPLITS.items():
        out_dir = img_dir  # 标签与图片同目录（Method 1 布局）
        xmls = sorted(xml_dir.glob("*.xml"))
        n_objs = n_miss_img = 0
        for xml_path in xmls:
            img_path = img_dir / (xml_path.stem + ".jpg")
            if not img_path.exists():
                n_miss_img += 1
                continue
            lines, n_skipped, unknown = convert_one_xml(xml_path)
            n_objs += len(lines)
            total_skipped += n_skipped
            all_unknown |= unknown
            (out_dir / (xml_path.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"[{split}] XML: {len(xmls)}, 无对应图片跳过: {n_miss_img}, 目标数: {n_objs}")
        total_imgs += len(xmls) - n_miss_img
        total_objs += n_objs
    print(f"\n合计: 图片 {total_imgs}, 目标 {total_objs}, 跳过目标 {total_skipped}")
    if all_unknown:
        print(f"警告: 出现未知类别名 {all_unknown}")
    print("输出: 各 split 的可见光图片目录（txt 与 jpg 同目录）")


if __name__ == "__main__":
    main()
