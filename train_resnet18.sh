job_name="resnet18"
OMP_NUM_THREADS=12 python train_resnet18.py \
    --batch_size 128 \
    --num_epochs 1 \
    --pretrained \
    --scheduled_unfreeze \
    --freeze | tee --append logs/${job_name}