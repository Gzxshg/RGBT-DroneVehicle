#!/usr/bin/env python3
"""Evaluate a DroneVehicle checkpoint.

Reports, in one run over the val set:
  - the standard 12 COCO bbox metrics (AP, AP50, AP75, AP-S/M/L, AR...)
  - per-class AP (AP@[.5:.95] for each of the 5 classes)
  - inference speed (batch=1 forward + postprocess, FPS and ms/img)

Example:
  python tools/eval_dv.py -m dn_dab_dino_deformable_detr --use_dn --contrastive \
      --two_stage --num_classes 5 --coco_path data/DroneVehicle/rgb \
      --checkpoint logs/dv_dino_r50_36e/rgb/checkpoint.pth --batch_size 1
"""
import json
import os
import sys
import time

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate
from main import build_model_main, get_args_parser

CLASSES = ('car', 'truck', 'freight_car', 'bus', 'van')
METRIC_NAMES = ['AP', 'AP50', 'AP75', 'APS', 'APM', 'APL',
                'AR1', 'AR10', 'AR100', 'ARS', 'ARM', 'ARL']


def per_class_ap(coco_eval, cat_ids):
    """Re-run COCOeval per category, return {name: AP}."""
    from pycocotools.cocoeval import COCOeval
    out = {}
    for cid, name in zip(cat_ids, CLASSES):
        e = COCOeval(coco_eval.cocoGt, coco_eval.cocoDt, 'bbox')
        e.params.catIds = [cid]
        e.evaluate()
        e.accumulate()
        e.summarize()
        out[name] = float(e.stats[0])
    return out


def measure_fps(model, postprocessors, dataset, device, warmup=50, num=500):
    """Batch-1 forward + postprocess timing over the val dataset."""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=utils.collate_fn)
    times = []
    with torch.no_grad():
        for i, (samples, targets) in enumerate(loader):
            if i >= warmup + num:
                break
            samples = samples.to(device)
            orig_size = torch.stack([t['orig_size'] for t in targets]).to(device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs, _ = model(samples)
            _ = postprocessors['bbox'](outputs, orig_size)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if i >= warmup:
                times.append(dt)
    n = len(times)
    total = sum(times)
    return {'fps': n / total if total > 0 else 0.0,
            'ms_per_img': 1000.0 * total / n if n else 0.0,
            'n_images': n}


def main():
    parser = get_args_parser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--fps_num', type=int, default=500)
    parser.add_argument('--fps_warmup', type=int, default=50)
    parser.add_argument('--metrics_out', type=str, default='',
                        help='where to write metrics json; default <output_dir>/eval_metrics.json')
    args = parser.parse_args()
    args.eval = True
    args.resume = ''
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    model, criterion, postprocessors = build_model_main(args)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    model.to(device)
    model.eval()
    criterion.to(device)

    dataset_val = build_dataset(image_set='val', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, args.batch_size, sampler=sampler_val, drop_last=False,
        collate_fn=utils.collate_fn, num_workers=args.num_workers)
    base_ds = get_coco_api_from_dataset(dataset_val)

    os.makedirs(args.output_dir, exist_ok=True)
    test_stats, coco_evaluator = evaluate(
        model, criterion, postprocessors, data_loader_val, base_ds, device,
        args.output_dir, args=args)

    metrics = {}
    coco12 = test_stats['coco_eval_bbox']
    metrics['coco'] = {n: float(v) for n, v in zip(METRIC_NAMES, coco12)}
    print('COCO bbox:', json.dumps(metrics['coco'], indent=2))

    cat_ids = coco_evaluator.coco_eval['bbox'].params.catIds
    metrics['per_class_ap'] = per_class_ap(coco_evaluator.coco_eval['bbox'],
                                           cat_ids)
    print('Per-class AP:', json.dumps(metrics['per_class_ap'], indent=2))

    metrics['speed'] = measure_fps(model, postprocessors, dataset_val, device,
                                   warmup=args.fps_warmup, num=args.fps_num)
    print('Speed:', json.dumps(metrics['speed'], indent=2))

    out = args.metrics_out or os.path.join(args.output_dir, 'eval_metrics.json')
    with open(out, 'w') as f:
        json.dump(metrics, f, indent=2)
    print('metrics written to', out)


if __name__ == '__main__':
    main()
