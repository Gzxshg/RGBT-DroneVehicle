#!/usr/bin/env python3
"""Convert DroneVehicle VOC-polygon (OBB) XML annotations to COCO JSON
with circumscribed horizontal bounding boxes (HBB).

For each modality (rgb/ir) and split (train/val) this writes a standard
COCO instances JSON:
  - bbox        : [x, y, w, h] circumscribed horizontal box of the quad
  - area        : w * h of the (clipped) HBB
  - segmentation: [[x1, y1, ..., x4, y4]] original OBB quad, kept so the
                  rotated-box information is preserved for future OBB work
  - category_id : 0..4  (car, truck, freight_car, bus, van)

It also creates the COCO-style directory layout with symlinks to the raw
image directories (no image data is copied):

  <out>/<mod>/train2017 -> <dv_root>/train/<imgdir>
  <out>/<mod>/val2017   -> <dv_root>/val/<imgdir>
  <out>/<mod>/annotations/instances_{train,val}2017.json

With --mini N an additional tiny dataset is written under <mini_out> where
both the train and val JSONs contain the same first N training images
(for overfit sanity tests). The original XML/txt annotations are never
modified.
"""
import argparse
import json
import os
import xml.etree.ElementTree as ET
from multiprocessing import Pool

CLASSES = ('car', 'truck', 'freight_car', 'bus', 'van')
CLASS_ALIAS = {
    'feright_car': 'freight_car', 'feright_car_': 'freight_car',
    'feright': 'freight_car', 'freight': 'freight_car',
    'freight_car': 'freight_car', 'freight_car_': 'freight_car',
}
CAT2ID = {c: i for i, c in enumerate(CLASSES)}

# modality -> (image subdir, label subdir); labels are per-modality on purpose:
# rgb experiments use {split}label (annotated on RGB), ir use {split}labelr.
MOD_DIRS = {
    'rgb': ('{s}img', '{s}label'),
    'ir': ('{s}imgr', '{s}labelr'),
}


def norm_class(name):
    n = name.strip().lower().replace(' ', '_').replace('-', '_')
    return CLASS_ALIAS.get(n, n)


def parse_xml(path):
    root = ET.parse(path).getroot()
    size = root.find('size')
    w, h = int(size.findtext('width')), int(size.findtext('height'))
    anns = []
    skipped = 0
    for obj in root.iter('object'):
        name = norm_class(obj.findtext('name', default=''))
        poly = obj.find('polygon')
        if name not in CAT2ID or poly is None:
            continue
        xs = [float(poly.findtext(f'x{i}')) for i in range(1, 5)]
        ys = [float(poly.findtext(f'y{i}')) for i in range(1, 5)]
        x0, x1 = max(min(xs), 0.0), min(max(xs), float(w))
        y0, y1 = max(min(ys), 0.0), min(max(ys), float(h))
        bw, bh = x1 - x0, y1 - y0
        if bw < 1.0 or bh < 1.0:
            skipped += 1
            continue
        quad = [c for xy in zip(xs, ys) for c in xy]
        anns.append({
            'category_id': CAT2ID[name],
            'bbox': [x0, y0, bw, bh],
            'area': bw * bh,
            'iscrowd': 0,
            'segmentation': [quad],
        })
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem, w, h, anns, skipped


def build_split(dv_root, split, mod, workers):
    img_sub, lab_sub = MOD_DIRS[mod]
    img_dir = os.path.join(dv_root, split, img_sub.format(s=split))
    lab_dir = os.path.join(dv_root, split, lab_sub.format(s=split))
    xmls = sorted(os.path.join(lab_dir, f)
                  for f in os.listdir(lab_dir) if f.endswith('.xml'))
    with Pool(workers) as pool:
        results = pool.map(parse_xml, xmls, chunksize=64)

    images, annotations = [], []
    ann_id = 0
    n_skip = 0
    for stem, w, h, anns, skipped in results:
        img_id = int(stem)
        images.append({'id': img_id, 'file_name': f'{stem}.jpg',
                       'width': w, 'height': h})
        for a in anns:
            a = dict(a)
            a['id'] = ann_id
            a['image_id'] = img_id
            annotations.append(a)
            ann_id += 1
        n_skip += skipped
    coco = {
        'info': {'description': f'DroneVehicle {split} {mod} (HBB from OBB quads)'},
        'categories': [{'id': i, 'name': c, 'supercategory': 'vehicle'}
                       for i, c in enumerate(CLASSES)],
        'images': images,
        'annotations': annotations,
    }
    print(f'[{split}/{mod}] images={len(images)} anns={len(annotations)} '
          f'degenerate_skipped={n_skip}')
    return img_dir, coco


def ensure_symlink(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        if os.path.realpath(dst) == os.path.realpath(src):
            return
        os.remove(dst) if os.path.islink(dst) else None
        if os.path.isdir(dst) and not os.path.islink(dst):
            raise RuntimeError(f'{dst} exists and is a real directory')
    os.symlink(src, dst)


def write_split(out_mod_dir, img_dir, split, coco):
    os.makedirs(out_mod_dir, exist_ok=True)
    link = os.path.join(out_mod_dir, f'{split}2017')
    ensure_symlink(img_dir, link)
    ann_dir = os.path.join(out_mod_dir, 'annotations')
    os.makedirs(ann_dir, exist_ok=True)
    with open(os.path.join(ann_dir, f'instances_{split}2017.json'), 'w') as f:
        json.dump(coco, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dv_root', default='/mnt/dataset/DroneVehicle')
    ap.add_argument('--out', default='data/DroneVehicle')
    ap.add_argument('--splits', nargs='+', default=['train', 'val'])
    ap.add_argument('--modalities', nargs='+', default=['rgb', 'ir'])
    ap.add_argument('--mini', type=int, default=0,
                    help='also write a mini overfit set with N train images')
    ap.add_argument('--mini_out', default='data/DroneVehicle_mini')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()

    for mod in args.modalities:
        mini_done = False
        for split in args.splits:
            img_dir, coco = build_split(args.dv_root, split, mod, args.workers)
            write_split(os.path.join(args.out, mod), img_dir, split, coco)

            if args.mini and split == 'train' and not mini_done:
                keep = sorted(img['id'] for img in coco['images'])[:args.mini]
                keep = set(keep)
                mini = {
                    'info': {'description': f'DroneVehicle mini {mod} (overfit)'},
                    'categories': coco['categories'],
                    'images': [im for im in coco['images'] if im['id'] in keep],
                    'annotations': [a for a in coco['annotations']
                                    if a['image_id'] in keep],
                }
                mini_mod = os.path.join(args.mini_out, mod)
                # train and val point at the same images/anns (overfit test)
                write_split(mini_mod, img_dir, 'train', mini)
                ensure_symlink(img_dir, os.path.join(mini_mod, 'val2017'))
                with open(os.path.join(mini_mod, 'annotations',
                                       'instances_val2017.json'), 'w') as f:
                    json.dump(mini, f)
                print(f'[mini/{mod}] {len(mini["images"])} images, '
                      f'{len(mini["annotations"])} anns -> {mini_mod}')
                mini_done = True


if __name__ == '__main__':
    main()
