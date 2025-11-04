#!/bin/bash

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate seggpt

# Run fine-tuning for all models sequentially
python finetune_remote_sensing_datasets.py --model ViT-B-16 --model-location ./models/checkpoints_remote_sensing
python finetune_remote_sensing_datasets.py --model ViT-B-32 --model-location ./models/checkpoints_remote_sensing
python finetune_remote_sensing_datasets.py --model ViT-L-14 --model-location ./models/checkpoints_remote_sensing

echo "All models finished!"

