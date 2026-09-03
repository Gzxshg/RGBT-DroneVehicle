#!/usr/bin/env python3
"""Build the "Matched" directory structure expected by C2Former's
DroneVehicleDataset from the raw official DroneVehicle release.

Raw layout (per split):
    {split}/{split}img/      RGB images  XXXXX.jpg
    {split}/{split}imgr/     IR images   XXXXX.jpg
    {split}/{split}labelr/   IR labels   XXXXX.xml (VOC-style, <polygon> quads)

Target layout (per split):
    {split}_total/{split}MatchedImg_total/
        XXXXX.jpg      -> symlink to RGB image
        XXXXX_tir.jpg  -> symlink to IR image
    {split}_total/{split}MatchedLabelTxtMVP_total/
        XXXXX_tir.txt  DOTA-format: x1 y1 x2 y2 x3 y3 x4 y4 class difficulty
"""
import os
import sys
import xml.etree.ElementTree as ET

CLASSES = ('car', 'truck', 'freight_car', 'bus', 'van')

# The raw release misspells "freight_car" in several ways.
CLASS_ALIAS = {
    'feright_car': 'freight_car', 'feright_car_': 'freight_car',
    'feright': 'freight_car', 'freight': 'freight_car',
    'freight_car': 'freight_car', 'freight_car_': 'freight_car',
}


def norm_class(name):
    n = name.strip().lower().replace(' ', '_').replace('-', '_')
    return CLASS_ALIAS.get(n, n)


def convert_xml(xml_path, txt_path):
    tree = ET.parse(xml_path)
    lines = []
    for obj in tree.getroot().iter('object'):
        name = norm_class(obj.findtext('name', default=''))
        if name not in CLASSES:
            print(f'WARN: unknown class {name!r} in {xml_path}, skipped')
            continue
        poly = obj.find('polygon')
        if poly is not None:
            coords = []
            for i in range(1, 5):
                coords.append(poly.findtext(f'x{i}'))
                coords.append(poly.findtext(f'y{i}'))
        else:
            box = obj.find('bndbox')
            if box is None:
                print(f'WARN: object without box in {xml_path}, skipped')
                continue
            xmin, ymin = box.findtext('xmin'), box.findtext('ymin')
            xmax, ymax = box.findtext('xmax'), box.findtext('ymax')
            coords = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
        difficult = obj.findtext('difficult', default='0')
        lines.append(' '.join(coords) + f' {name} {difficult}')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + ('\n' if lines else ''))
    return len(lines)


def process_split(root, split):
    img_dir = os.path.join(root, split, f'{split}img')
    imgr_dir = os.path.join(root, split, f'{split}imgr')
    labelr_dir = os.path.join(root, split, f'{split}labelr')
    out_img_dir = os.path.join(root, f'{split}_total', f'{split}MatchedImg_total')
    out_lbl_dir = os.path.join(root, f'{split}_total', f'{split}MatchedLabelTxtMVP_total')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.jpg'))
    n_boxes = 0
    missing = []
    for img_id in ids:
        rgb = os.path.join(img_dir, img_id + '.jpg')
        tir = os.path.join(imgr_dir, img_id + '.jpg')
        xml = os.path.join(labelr_dir, img_id + '.xml')
        if not (os.path.exists(tir) and os.path.exists(xml)):
            missing.append(img_id)
            continue
        for src, dst in ((rgb, img_id + '.jpg'), (tir, img_id + '_tir.jpg')):
            link = os.path.join(out_img_dir, dst)
            if not os.path.exists(link):
                os.symlink(src, link)
        n_boxes += convert_xml(xml, os.path.join(out_lbl_dir, img_id + '_tir.txt'))
    print(f'[{split}] pairs: {len(ids) - len(missing)}/{len(ids)}, boxes: {n_boxes}, missing: {len(missing)}')
    if missing:
        print(f'  missing ids: {missing[:10]}{"..." if len(missing) > 10 else ""}')


if __name__ == '__main__':
    root = os.path.dirname(os.path.abspath(__file__))
    for split in sys.argv[1:] or ['train', 'val', 'test']:
        process_split(root, split)
