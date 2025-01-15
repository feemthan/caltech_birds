#!/bin/bash
job_name="resnet50dist"
OMP_NUM_THREADS=12 python train_resnet50_dist.py \
    --batch_size 128 \
    --num_augmentations 4 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --pretrained | tee -a logs/${job_name}