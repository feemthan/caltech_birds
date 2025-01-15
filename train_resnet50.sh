job_name="resnet50"
OMP_NUM_THREADS=12 python train_resnet50.py \
    --batch_size 32 \
    --num_epochs 400 \
    --pretrained \
    --scheduled_unfreeze \
    --freeze | tee -a logs/${job_name}
