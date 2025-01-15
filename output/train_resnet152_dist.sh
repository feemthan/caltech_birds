#!/bin/bash

job_name="resnet152dist"
config_file="config.json"

# Create a new config file (overwriting any existing file)
cat > ${config_file} << EOF
{
    "root_dir": "/home/faheem/Workspace/caltech_birds/images/",
    "train_txt": "/home/faheem/Workspace/caltech_birds/lists/train.txt",
    "test_txt": "/home/faheem/Workspace/caltech_birds/lists/test.txt",
    "class_names": "/home/faheem/Workspace/caltech_birds/lists/classes.txt",
    "batch_size": 60,
    "num_augmentations": 5,
    "num_epochs": 15,
    "learning_rate": 1e-4,
    "pretrained": true,
    "freeze": false,
    "scheduled_unfreeze": false
}
EOF

# Run the Python script with the config file
OMP_NUM_THREADS=12 python train_resnet152_dist.py --config ${config_file} | tee -a logs/${job_name}
