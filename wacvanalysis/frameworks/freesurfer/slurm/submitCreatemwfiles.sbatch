#!/bin/bash
#SBATCH --job-name=mwfiles
#SBATCH --time=10:00:00
#SBATCH -p qTRDHM
#SBATCH -A psy53c17
#SBATCH --output=jobs/out%a_%A.out
#SBATCH --error=jobs/err%a_%A.err
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1

. /data/users2/washbee/anaconda3latest/bin/activate wacvanalysis

SUBJECTS_DIR="/data/users2/washbee/speedrun/cortexode-data-rp/test/"
OUTPUT_DIR="../mwremoved"
SUBJECT_LIST="../../test_subs.txt" # File containing the list of subject IDs, one per line

# Read the subject ID for the current task
SUBJECT_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)

python ../../freesurfer/medial_wall_util.py --subjects_dir $SUBJECTS_DIR --subject_id $SUBJECT_ID --output_dir $OUTPUT_DIR
