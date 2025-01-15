job_name="resnet152"
OMP_NUM_THREADS=12 python train_resnet152.py \
    --batch_size 64 \
    --num_epochs 4 \
    --pretrained \
    --scheduled_unfreeze \
    --freeze | tee -a logs/${job_name}
