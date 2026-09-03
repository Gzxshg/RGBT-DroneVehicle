#!/bin/bash
# DroneVehicle pipeline supervisor: wait for overfit runs -> gate on overfit AP
# -> launch full training (rgb then ir) -> auto-eval. Detached via setsid/nohup.
# Status is appended to results/pipeline_status.txt.

set -u
cd /home/gzx26/dataY/DN-DETR
PY=/mnt/dataX/gzx26/miniconda3/envs/c2former/bin/python
STATUS=results/pipeline_status.txt
mkdir -p results logs/dv_dino_r50_36e

log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

overfit_ap() {  # $1 = modality; prints last-epoch test AP
    $PY - "$1" <<'EOF'
import json, sys
try:
    lines = open(f'logs/dv_overfit/{sys.argv[1]}/log.txt').read().strip().split('\n')
    d = json.loads(lines[-1])
    print(d['test_coco_eval_bbox'][0])
except Exception:
    print(-1)
EOF
}

pick_gpus() {  # prints 4 least-used GPU ids, comma-separated
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      | awk '{print $1, NR-1}' | sort -n | head -4 | cut -d' ' -f2 | paste -sd,
}

run_full() {  # $1 = modality, $2 = master_port
    local mod=$1 port=$2
    local gpus
    gpus=$(pick_gpus)
    log "full train $mod starting on GPUs [$gpus]"
    CUDA_VISIBLE_DEVICES=$gpus nohup $PY -m torch.distributed.launch \
      --nproc_per_node=4 --master_port=$port \
      main.py -m dn_dab_dino_deformable_detr --use_dn --contrastive --two_stage \
      --num_classes 5 --coco_path data/DroneVehicle/$mod \
      --output_dir logs/dv_dino_r50_36e/$mod \
      --batch_size 2 --epochs 36 --lr_drop 30 \
      --num_workers 4 --save_checkpoint_interval 10 \
      > logs_dv_full_$mod.out 2>&1
    local rc=$?
    log "full train $mod finished, exit=$rc"
    if [ $rc -eq 0 ] && [ -f logs/dv_dino_r50_36e/$mod/checkpoint.pth ]; then
        log "eval $mod final checkpoint"
        CUDA_VISIBLE_DEVICES=$(pick_gpus | cut -d, -f1) nohup $PY tools/eval_dv.py \
          -m dn_dab_dino_deformable_detr --use_dn --contrastive --two_stage \
          --num_classes 5 --coco_path data/DroneVehicle/$mod \
          --checkpoint logs/dv_dino_r50_36e/$mod/checkpoint.pth \
          --output_dir logs/dv_dino_r50_36e/$mod --batch_size 1 --num_workers 4 \
          >> logs_dv_full_$mod.out 2>&1
        log "eval $mod done, exit=$? (metrics: logs/dv_dino_r50_36e/$mod/eval_metrics.json)"
    else
        log "SKIP eval $mod (training failed or no checkpoint)"
    fi
}

log "supervisor started, waiting for overfit runs..."
while pgrep -f 'logs/dv_overfit/' > /dev/null; do sleep 60; done
sleep 30
log "overfit processes finished"

for mod in rgb ir; do
    ep=$(wc -l < logs/dv_overfit/$mod/log.txt 2>/dev/null || echo 0)
    ap=$(overfit_ap $mod)
    log "overfit $mod: epochs=$ep final_test_AP=$ap"
    pass=$($PY -c "print(1 if float('$ap') >= 0.80 else 0)")
    if [ "$pass" = "1" ]; then
        log "GATE PASS $mod -> launch full training"
    else
        log "GATE FAIL $mod -> full training NOT launched"
    fi
done

for mod in rgb ir; do
    ap=$(overfit_ap $mod)
    pass=$($PY -c "print(1 if float('$ap') >= 0.80 else 0)")
    if [ "$pass" = "1" ]; then
        port=29901; [ "$mod" = "ir" ] && port=29902
        run_full $mod $port
    fi
done

log "supervisor done"
