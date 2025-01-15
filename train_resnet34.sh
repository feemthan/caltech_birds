job_name="resnet34"
OMP_NUM_THREADS=12 python train_resnet34.py \
    --batch_size 128 \
    --num_epochs 100 \
    --freeze | tee -a logs/${job_name}
