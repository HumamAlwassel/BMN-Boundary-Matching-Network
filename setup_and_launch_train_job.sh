#!/bin/bash --login

NUM_RUNS=6
START_RUN_ID=0
TEMPORAL_SCALE=100
CONCAT_OR_SUM_GLOBAL_MAX_FEATURE=0 # 0: do not include global features; 1: concat global features; 2: sum global features
N_GPU=1
BATCH_SIZE=$(( 16 * N_GPU ))
TRAINING_LR=`bc -l <<< "0.0005 * $N_GPU" | sed -e "/\./ s/0*\s*$//" -e 's/^\./0./'`
TRAIN_EPOCHS=10
OUTPUT_ROOT=/ibex/scratch/alwassha/pytorch-experiments/bmn/activitynet_feature_e2e-video/interpolated_${TEMPORAL_SCALE}/

for FEATURE_TYPE in \
r2plus1d-18_features_one-head_fc-only-0.004_model_5 \
r2plus1d-18_features_one-head_0.001-0.001-0.001-0.001-0.008_model_5 \
r2plus1d-18_features_one-head_0.0001-0.0001-0.0001-0.0001-0.004_model_5 \
r2plus1d-18_features_two-heads-A-noA-1.0-1.0_0.001-0.001-0.001-0.001-0.004_model_7 \
r2plus1d-18_features_two-heads-A-noA-1.0-1.0_0.0001-0.0001-0.0001-0.0001-0.004_model_6 \
r2plus1d-18_features_two-heads-A-noA-with-global-avg-features-1.0-1.0_0.0001-0.0001-0.0001-0.0001-0.002_model_6 \
r2plus1d-18_features_two-heads-A-noA-with-global-max-features-1.0-1.0_0.0001-0.0001-0.0001-0.0001-0.002_model_6 \
r2plus1d-18_features_two-heads-A-noA-both-with-global-max-features-1.0-1.0_0.0001-0.0001-0.0001-0.0001-0.002_model_6 \
r2plus1d-34_features_one-head_fc-only-0.004_model_5 \
r2plus1d-34_features_one-head_0.0001-0.0001-0.0001-0.0001-0.002_model_5 \
r2plus1d-34_features_two-heads-A-noA-1.0-1.0_0.0001-0.0001-0.0001-0.0001-0.004_model_5 \
r2plus1d-34_features_two-heads-A-noA-with-global-avg-features-1.0-1.0_0.0001-0.0001-0.0001-0.0001-0.004_model_5 \
r2plus1d-34_features_two-heads-A-noA-with-global-max-features-1.0-1.0_0.0001-0.0001-0.0001-0.0001-0.002_model_5 \
r2plus1d-34_features_two-heads-A-noA-both-with-global-max-features-1.0-1.0_0.0001-0.0001-0.0001-0.0001-0.002_model_7 
do
        if [[ CONCAT_OR_SUM_GLOBAL_MAX_FEATURE -eq 1 ]]; then
                FEATURE_TYPE=${FEATURE_TYPE}_interpolated_${TEMPORAL_SCALE}_concat_global_max_feature
                FEAT_DIM=1024
        elif [[ CONCAT_OR_SUM_GLOBAL_MAX_FEATURE -eq 2 ]]; then
                FEATURE_TYPE=${FEATURE_TYPE}_interpolated_${TEMPORAL_SCALE}_sum_global_max_feature
                FEAT_DIM=512
        else
                FEATURE_TYPE=${FEATURE_TYPE}_interpolated_${TEMPORAL_SCALE}
                FEAT_DIM=512
        fi

        if [[ "${FEATURE_TYPE}" == *-152* ]]; then
                FEAT_DIM=$(( FEAT_DIM * 4 ))
        fi

        FEATURE_PATH=/ibex/scratch/alwassha/activitynet_feature_e2e-video/activitynet_feature_e2e-video/interpolated_${TEMPORAL_SCALE}/${FEATURE_TYPE}.h5
        for i in $( seq $START_RUN_ID $(( START_RUN_ID + NUM_RUNS - 1 )) )
        do
                RUN_ID=run_${i}
                OUTPUT_PATH=${OUTPUT_ROOT}/${FEATURE_TYPE}/${TRAINING_LR}/${RUN_ID}
                JOB_NAME=BMN-ANET-${FEATURE_TYPE}-${TRAINING_LR}-${RUN_ID}

                mkdir -p $OUTPUT_PATH
                echo $JOB_NAME

                sbatch --gres=gpu:v100:${N_GPU} --job-name=${JOB_NAME} \
                --export=ALL,TEMPORAL_SCALE=$TEMPORAL_SCALE,FEATURE_TYPE=$FEATURE_TYPE,FEATURE_PATH=$FEATURE_PATH,FEAT_DIM=$FEAT_DIM,N_GPU=$N_GPU,BATCH_SIZE=$BATCH_SIZE,TRAIN_EPOCHS=$TRAIN_EPOCHS,TRAINING_LR=$TRAINING_LR,OUTPUT_PATH=$OUTPUT_PATH,JOB_NAME=$JOB_NAME \
                slurm_train_bmn.sh
        done
done
