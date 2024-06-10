#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4g
#SBATCH -p qTRD
#SBATCH -t 1-00:00
#SBATCH -J v2ccp
#SBATCH -e jobs/error%A_%a.err
#SBATCH -o jobs/out%A_%a.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe

sleep 5s

source $FREESURFER_HOME/SetUpFreeSurfer.sh
fold='train'
export SUBJECTS_DIR=/data/users2/washbee/speedrun/cortexode-data-rp/${fold}
export SAVE_DIR=/data/users2/washbee/speedrun/cortexode-data-rp/${fold}/${fold}_v2cc
export SUBJ_LIST=/data/users2/washbee/speedrun/cortexode-data-rp/${fold}/all_subjects.txt
original_permissions=$(stat -c "%a" $SUBJECTS_DIR)

mkdir -p $SAVE_DIR


# Read the specific line from $SUBJ_LIST based on $SLURM_ARRAY_TASK_ID
sub=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJ_LIST)

# Log the subject being processed
echo "Processing subject: $sub"

# Call the preprocessing script with the subject as an argument
chmod 755 preprocv2cc.sh
./preprocv2cc.sh $sub

sleep 10s
