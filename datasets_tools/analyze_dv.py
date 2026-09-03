#!/usr/bin/env python3
"""Check DroneVehicle structure / RGB-IR pairing and compute OBB statistics.

For each split (train/val) and each annotation source (rgb: {split}label,
ir: {split}labelr) this script parses the VOC-style XML files with
<polygon> quadrilateral annotations and reports:
  - per-image object count
  - OBB long-edge / short-edge length (from quad edges)
  - OBB polygon area (shoelace)
  - OBB long-edge angle in degrees, mapped to [-90, 90)
  - per-class instance counts

Outputs a text report and histogram PNGs under the output directory.
"""
import argparse
import math
import os
import xml.etree.ElementTree as ET
from collections import Counter
from multiprocessing import Pool

import numpy as np

CLASSES = ('car', 'truck', 'freight_car', 'bus', 'van')
CLASS_ALIAS = {
    'feright_car': 'freight_car', 'feright_car_': 'freight_car',
    'feright': 'freight_car', 'freight': 'freight_car',
    'freight_car': 'freight_car', 'freight_car_': 'freight_car',
}


def norm_class(name):
    n = name.strip().lower().replace(' ', '_').replace('-', '_')
    return CLASS_ALIAS.get(n, n)


def parse_xml(path):
    """Return (width, height, [(cls, (x1..y4)), ...]) from a VOC polygon XML."""
    root = ET.parse(path).getroot()
    size = root.find('size')
    w = int(size.findtext('width'))
    h = int(size.findtext('height'))
    objs = []
    unknown = []
    for obj in root.iter('object'):
        name = norm_class(obj.findtext('name', default=''))
        poly = obj.find('polygon')
        if poly is None:
            continue
        pts = tuple(float(poly.findtext(f'{ax}{i}')) for i in range(1, 5)
                    for ax in ('x', 'y'))
        if name not in CLASSES:
            unknown.append(name)
            continue
        objs.append((name, pts))
    return w, h, objs, unknown


def quad_measure(pts):
    """(long_edge, short_edge, area, angle_deg) of a quadrilateral.

    angle_deg: orientation of the longer of the two adjacent edge
    directions, mapped into [-90, 90).
    """
    xs = pts[0::2]
    ys = pts[1::2]
    e1 = (xs[1] - xs[0], ys[1] - ys[0])
    e2 = (xs[2] - xs[1], ys[2] - ys[1])
    l1 = math.hypot(*e1)
    l2 = math.hypot(*e2)
    long_e, short_e = max(l1, l2), min(l1, l2)
    area = 0.5 * abs(sum(xs[i] * ys[(i + 1) % 4] - xs[(i + 1) % 4] * ys[i]
                         for i in range(4)))
    vec = e1 if l1 >= l2 else e2
    ang = math.degrees(math.atan2(vec[1], vec[0]))
    ang = (ang + 90.0) % 180.0 - 90.0
    return long_e, short_e, area, ang


def worker(args):
    path, = args
    try:
        return os.path.basename(path), parse_xml(path)
    except Exception as e:
        return os.path.basename(path), ('ERR', str(e))


def describe(name, arr, fmt='{:.1f}'):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return f'{name}: empty'
    q = np.percentile(arr, [5, 25, 50, 75, 95])
    return (f'{name}: n={arr.size} mean={fmt.format(arr.mean())} '
            f'min={fmt.format(arr.min())} p5={fmt.format(q[0])} '
            f'p25={fmt.format(q[1])} median={fmt.format(q[2])} '
            f'p75={fmt.format(q[3])} p95={fmt.format(q[4])} '
            f'max={fmt.format(arr.max())}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dv_root', default='/mnt/dataset/DroneVehicle')
    ap.add_argument('--out', default='results/dv_stats')
    ap.add_argument('--splits', nargs='+', default=['train', 'val'])
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lines = []

    def log(s=''):
        print(s)
        lines.append(s)

    log(f'DroneVehicle root: {args.dv_root}')
    log('=' * 70)

    # ---- structure & pairing check ----
    for split in args.splits:
        img_dir = os.path.join(args.dv_root, split, f'{split}img')
        imgr_dir = os.path.join(args.dv_root, split, f'{split}imgr')
        lab_dir = os.path.join(args.dv_root, split, f'{split}label')
        labr_dir = os.path.join(args.dv_root, split, f'{split}labelr')
        imgs = {f[:-4] for f in os.listdir(img_dir) if f.endswith('.jpg')}
        imgrs = {f[:-4] for f in os.listdir(imgr_dir) if f.endswith('.jpg')}
        labs = {f[:-4] for f in os.listdir(lab_dir) if f.endswith('.xml')}
        labrs = {f[:-4] for f in os.listdir(labr_dir) if f.endswith('.xml')}
        log(f'[{split}] rgb_img={len(imgs)} ir_img={len(imgrs)} '
            f'rgb_xml={len(labs)} ir_xml={len(labrs)}')
        log(f'  RGB without IR pair: {len(imgs - imgrs)}, '
            f'IR without RGB pair: {len(imgrs - imgs)}')
        log(f'  img without rgb_xml: {len(imgs - labs)}, '
            f'img without ir_xml: {len(imgs - labrs)}')
    log('=' * 70)

    # ---- per split/modality stats ----
    for split in args.splits:
        for mod, lab_name in (('rgb', f'{split}label'), ('ir', f'{split}labelr')):
            lab_dir = os.path.join(args.dv_root, split, lab_name)
            paths = sorted(os.path.join(lab_dir, f)
                           for f in os.listdir(lab_dir) if f.endswith('.xml'))
            with Pool(args.workers) as pool:
                results = pool.map(worker, [(p,) for p in paths], chunksize=64)

            cls_counter = Counter()
            unknown_counter = Counter()
            per_img_cnt = []
            longs, shorts, areas, angs = [], [], [], []
            sizes = Counter()
            errors = []
            zero_obj = 0
            for stem, res in results:
                if isinstance(res[0], str) and res[0] == 'ERR':
                    errors.append((stem, res[1]))
                    continue
                w, h, objs, unknown = res
                sizes[(w, h)] += 1
                per_img_cnt.append(len(objs))
                if len(objs) == 0:
                    zero_obj += 1
                for name in unknown:
                    unknown_counter[name] += 1
                for name, pts in objs:
                    cls_counter[name] += 1
                    le, se, ar, an = quad_measure(pts)
                    longs.append(le)
                    shorts.append(se)
                    areas.append(ar)
                    angs.append(an)

            tag = f'{split}/{mod}'
            log(f'--- {tag} ---')
            log(f'images parsed: {len(per_img_cnt)}, errors: {len(errors)}, '
                f'zero-object images: {zero_obj} '
                f'({100.0 * zero_obj / max(len(per_img_cnt), 1):.1f}%)')
            log(f'image sizes: {dict(sizes.most_common(3))}')
            log(f'total instances: {sum(cls_counter.values())}')
            for c in CLASSES:
                log(f'  {c:12s}: {cls_counter[c]} '
                    f'({100.0 * cls_counter[c] / max(sum(cls_counter.values()), 1):.2f}%)')
            if unknown_counter:
                log(f'  UNKNOWN (skipped): {dict(unknown_counter)}')
            per_img_arr = np.asarray(per_img_cnt, dtype=np.float64)
            log(describe('objects per image', per_img_arr))
            log(describe('OBB long edge (px)', longs))
            log(describe('OBB short edge (px)', shorts))
            log(describe('OBB area (px^2)', areas))
            log(describe('OBB long-edge angle (deg)', angs))
            # sqrt(area) ~ COCO-style size proxy
            sqrt_area = np.sqrt(np.asarray(areas)) if areas else np.asarray([])
            if sqrt_area.size:
                small = np.mean(sqrt_area < 32) * 100
                medium = np.mean((sqrt_area >= 32) & (sqrt_area < 96)) * 100
                large = np.mean(sqrt_area >= 96) * 100
                log(f'COCO size buckets (by OBB sqrt-area): '
                    f'small={small:.1f}% medium={medium:.1f}% large={large:.1f}%')
            log('')

            # ---- histograms ----
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            fig.suptitle(f'DroneVehicle {tag}')
            ax = axes[0, 0]
            ax.hist(per_img_arr, bins=range(0, int(per_img_arr.max()) + 2),
                    edgecolor='k')
            ax.set_title('objects per image')
            ax.set_yscale('log')
            axes[0, 1].hist(longs, bins=100, edgecolor='k')
            axes[0, 1].set_title('OBB long edge (px)')
            axes[0, 2].hist(shorts, bins=100, edgecolor='k')
            axes[0, 2].set_title('OBB short edge (px)')
            axes[1, 0].hist(np.log10(np.maximum(areas, 1e-6)), bins=100,
                            edgecolor='k')
            axes[1, 0].set_title('OBB area log10(px^2)')
            axes[1, 1].hist(angs, bins=np.linspace(-90, 90, 73), edgecolor='k')
            axes[1, 1].set_title('long-edge angle (deg)')
            cnts = [cls_counter[c] for c in CLASSES]
            axes[1, 2].bar(CLASSES, cnts)
            axes[1, 2].set_title('class counts')
            axes[1, 2].set_yscale('log')
            for a in axes.flat:
                a.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, f'hist_{split}_{mod}.png'),
                        dpi=110)
            plt.close(fig)

    with open(os.path.join(args.out, 'report.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nReport written to {os.path.join(args.out, "report.txt")}')


if __name__ == '__main__':
    main()
