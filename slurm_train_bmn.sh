#!/bin/bash --login
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%A-%3a.out
#SBATCH --error=logs/%x-%A-%3a.err
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=30G
#SBATCH --mail-type=FAIL,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --mail-user=humam.alwassel@kaust.edu.sa
#SBATCH -A conf-gpu-2020.11.23


set -ex

hostname
nvidia-smi

conda activate gtad

python main.py --mode train --n_gpu $N_GPU --batch_size $BATCH_SIZE --train_epochs $TRAIN_EPOCHS --training_lr $TRAINING_LR  --output $OUTPUT_PATH --temporal_scale $TEMPORAL_SCALE --feature_path $FEATURE_PATH --feat_dim $FEAT_DIM
python main.py --mode inference --output $OUTPUT_PATH --temporal_scale $TEMPORAL_SCALE --feature_path $FEATURE_PATH --feat_dim $FEAT_DIM
python detection_result_generate_cuhk_share.py --output $OUTPUT_PATH
