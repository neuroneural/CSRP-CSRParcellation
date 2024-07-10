#!/bin/bash
#SBATCH -p qTRDHM
#SBATCH -A psy53c17
#SBATCH --job-name=alyzsub
#SBATCH --output=jobs/out%a_%A.out
#SBATCH --error=jobs/err%a_%A.err
#SBATCH --time=10:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe

#start array at 1
source /data/users2/washbee/anaconda3latest/bin/activate wacvanalysis

# File containing the list of subject IDs
#SUBJ_FILE="test_subs.txt"
SUBJ_FILE="../../test_subs.txt"
# Get the subject ID for this task
subj_id=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJ_FILE)
#start at 1 
# Paths
csv_file_path="../csvs/cortexode_measures.csv"
framework_name="cortexode"
gt_subject_base_path="/data/users2/washbee/speedrun/cortexode-data-rp/test"
proj_subject_base_path="/data/users2/washbee/CortexODE-CSRFusionNet/wacvanalysis/frameworks/cortexode/alignment"

# Run the Python script
python ../../../analyze_subject_csrf.py $subj_id $csv_file_path $framework_name $gt_subject_base_path $proj_subject_base_path

