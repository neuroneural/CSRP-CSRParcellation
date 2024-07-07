#!/bin/bash

#SBATCH --job-name=mwrm
#SBATCH --time=10:00:00
#SBATCH -p qTRDHM
#SBATCH -A psy53c17
#SBATCH -c 2
#SBATCH --mem=5g
#SBATCH --output=jobs/out%a_%A.out
#SBATCH --error=jobs/err%a_%A.err
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1

. /data/users2/washbee/anaconda3latest/bin/activate wacvanalysis

# Define paths and parameters
SUBJECTS_DIR="/data/users2/washbee/speedrun/cortexode-data-rp/test"
PROJECT="cortexode"
PROJECT_GT_BASE_PATH="/data/users2/washbee/speedrun/cortexode-data-rp/test"
PROJECT_PRED_BASE_PATH="../result"
OUTPUT_FOLDER="../alignment"

# Read subject ID corresponding to the current SLURM array task ID
SUBJECTS_FILE="../../test_subs.txt"
SUBJECT_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECTS_FILE)

# Execute Python scripts for each hemisphere and surface type
python ../scripts/align_mwrm.py \
    --subjects_dir $SUBJECTS_DIR \
    --subject_id $SUBJECT_ID \
    --hemi "lh" \
    --surfType "pial" \
    --project $PROJECT \
    --project_gt_base_path $PROJECT_GT_BASE_PATH \
    --project_pred_base_path $PROJECT_PRED_BASE_PATH \
    --output_folder $OUTPUT_FOLDER

python ../scripts/align_mwrm.py \
    --subjects_dir $SUBJECTS_DIR \
    --subject_id $SUBJECT_ID \
    --hemi "lh" \
    --surfType "white" \
    --project $PROJECT \
    --project_gt_base_path $PROJECT_GT_BASE_PATH \
    --project_pred_base_path $PROJECT_PRED_BASE_PATH \
    --output_folder $OUTPUT_FOLDER

python ../scripts/align_mwrm.py \
    --subjects_dir $SUBJECTS_DIR \
    --subject_id $SUBJECT_ID \
    --hemi "rh" \
    --surfType "pial" \
    --project $PROJECT \
    --project_gt_base_path $PROJECT_GT_BASE_PATH \
    --project_pred_base_path $PROJECT_PRED_BASE_PATH \
    --output_folder $OUTPUT_FOLDER

python ../scripts/align_mwrm.py \
    --subjects_dir $SUBJECTS_DIR \
    --subject_id $SUBJECT_ID \
    --hemi "rh" \
    --surfType "white" \
    --project $PROJECT \
    --project_gt_base_path $PROJECT_GT_BASE_PATH \
    --project_pred_base_path $PROJECT_PRED_BASE_PATH \
    --output_folder $OUTPUT_FOLDER

