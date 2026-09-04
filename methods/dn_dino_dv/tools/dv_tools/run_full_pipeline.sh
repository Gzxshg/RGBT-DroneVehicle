#!/bin/bash
# DroneVehicle DN-DINO full training pipeline (RGB then IR), launched after the
# 20-image overfit gate passed. Train 36e on 4 GPUs -> auto eval (COCO 12
# metrics + per-class AP + FPS via tools/eval_dv.py). Detached via setsid.
# Status appended to results/pipeline_status.txt.

set -u
cd /home/gzx26/dataY/YOLOv11-RGBT/methods/dn_dino_dv
PY=/mnt/dataX/gzx26/miniconda3/envs/c2former/bin/python
STATUS=results/pipeline_status.txt
mkdir -p results logs/dv_dino_r50_36e

log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

pick_gpus() {  # prints 4 least-used GPU ids, comma-separated
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      | awk '{print $1, NR-1}' | sort -n | head -4 | cut -d' ' -f2 | paste -sd,
}

MODEL_ARGS="-m dn_dab_dino_deformable_detr --use_dn --contrastive --two_stage --num_classes 5"

run_one() {  # $1 = modality (rgb|ir), $2 = master_port
    local mod=$1 port=$2
    local gpus
    gpus=$(pick_gpus)
    log "full train $mod starting on GPUs [$gpus]"
    CUDA_VISIBLE_DEVICES=$gpus $PY -m torch.distributed.launch \
      --nproc_per_node=4 --master_port=$port \
      main.py $MODEL_ARGS \
      --coco_path data/DroneVehicle/$mod \
      --output_dir logs/dv_dino_r50_36e/$mod \
      --batch_size 2 --epochs 36 --lr_drop 30 \
      --num_workers 4 --save_checkpoint_interval 10 \
      > logs_dv_full_$mod.out 2>&1
    local rc=$?
    log "full train $mod finished, exit=$rc"
    if [ $rc -eq 0 ] && [ -f logs/dv_dino_r50_36e/$mod/checkpoint.pth ]; then
        log "eval $mod final checkpoint"
        CUDA_VISIBLE_DEVICES=$(pick_gpus | cut -d, -f1) $PY tools/eval_dv.py \
          $MODEL_ARGS --coco_path data/DroneVehicle/$mod \
          --checkpoint logs/dv_dino_r50_36e/$mod/checkpoint.pth \
          --output_dir logs/dv_dino_r50_36e/$mod --batch_size 1 --num_workers 4 \
          >> logs_dv_full_$mod.out 2>&1
        log "eval $mod done, exit=$? (metrics: logs/dv_dino_r50_36e/$mod/eval_metrics.json)"
    else
        log "SKIP eval $mod (training failed or no checkpoint)"
    fi
}

log "full pipeline started (rgb -> ir, sequential, 4 GPUs each)"
run_one rgb 29901
run_one ir 29902
log "full pipeline done"
