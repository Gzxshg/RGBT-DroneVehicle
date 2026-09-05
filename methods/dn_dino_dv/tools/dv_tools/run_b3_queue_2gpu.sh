#!/bin/bash
# B3 queue manager (2-GPU mode). Only GPUs 0-3 are usable on this box.
# Each run (b3a, b3b) starts as soon as >=2 of GPUs 0-3 are free (<2 GiB used),
# 2 GPUs x bs=2 per run, 36 epochs, lr_drop 30, save_interval 5 — all other
# hyperparameters identical to B2. Runs may overlap when 4 GPUs are free;
# a flock-protected claims file prevents two chains from grabbing the same GPUs.
# After each training, the final checkpoint is evaluated (tools/eval_dv.py).
# Does NOT touch logs/dv_dino_r50_36e (B2 frozen). Status: results/pipeline_status_b3.txt.

set -u
cd /home/gzx26/dataY/YOLOv11-RGBT/methods/dn_dino_dv
PY=/mnt/dataX/gzx26/miniconda3/envs/c2former/bin/python
STATUS=results/pipeline_status_b3.txt
CLAIMS=results/.b3_claims
LOCK=results/.b3_claims.lock
mkdir -p results
touch "$CLAIMS"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

MODEL_ARGS="-m dn_dab_dino_deformable_detr --use_dn --contrastive --two_stage --num_classes 5"

pick_and_claim() {  # $1 = run name; prints "g0,g1" and claims atomically, or prints nothing
    flock "$LOCK" bash -c '
        name=$1
        claimed=$(cut -d: -f2 results/.b3_claims 2>/dev/null | tr "," "\n" | grep -v "^$" || true)
        avail=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                | awk -F", " "\$1<4 && \$2<2048{print \$1}" | { grep -vxF "$claimed" || true; })
        n=$(echo "$avail" | grep -c . || true)
        if [ "$n" -ge 2 ]; then
            gpus=$(echo "$avail" | head -2 | paste -sd,)
            echo "$name:$gpus" >> results/.b3_claims
            echo "$gpus"
        fi' _ "$1"
}

release() { sed -i "/^$1:/d" "$CLAIMS"; }

chain() {  # $1 = run name (b3a|b3b), $2 = master_port, $3 = extra model/data args
    local name=$1 port=$2 extra=$3
    local gpus=""
    log "chain $name: waiting for 2 free GPUs among 0-3"
    while [ -z "$gpus" ]; do
        gpus=$(pick_and_claim "$name")
        [ -z "$gpus" ] && sleep 120
    done
    log "full train $name starting on GPUs [$gpus] (2-GPU, bs=2/GPU)"
    local ckpt=logs/dv_${name}_r50_36e/checkpoint.pth
    local resume=""
    [ -f "$ckpt" ] && resume="--resume $ckpt" && log "chain $name: resuming from $ckpt"
    CUDA_VISIBLE_DEVICES=$gpus $PY -m torch.distributed.launch \
      --nproc_per_node=2 --master_port=$port \
      main.py $MODEL_ARGS $extra $resume \
      --coco_path data/DroneVehicle/$name \
      --output_dir logs/dv_${name}_r50_36e \
      --batch_size 2 --epochs 36 --lr_drop 30 \
      --num_workers 4 --save_checkpoint_interval 5 \
      > logs_dv_full_$name.out 2>&1
    local rc=$?
    log "full train $name finished, exit=$rc"
    if [ $rc -eq 0 ] && [ -f logs/dv_${name}_r50_36e/checkpoint.pth ]; then
        local g1=${gpus%%,*}
        log "eval $name final checkpoint on GPU $g1"
        CUDA_VISIBLE_DEVICES=$g1 $PY tools/eval_dv.py \
          $MODEL_ARGS $extra --coco_path data/DroneVehicle/$name \
          --checkpoint logs/dv_${name}_r50_36e/checkpoint.pth \
          --output_dir logs/dv_${name}_r50_36e --batch_size 1 --num_workers 4 \
          >> logs_dv_full_$name.out 2>&1
        log "eval $name done, exit=$? (metrics: logs/dv_${name}_r50_36e/eval_metrics.json)"
    else
        log "TRAIN FAILED $name (exit=$rc) — eval skipped, claim released"
    fi
    release "$name"
}

log "B3 queue manager started (2-GPU mode, GPUs 0-3)"
chain b3a 29921 "" &
chain b3b 29922 "--pair_ir --fusion naive_concat1x1" &
wait
log "B3 queue done"
