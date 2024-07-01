#!/bin/bash
#SBATCH -p qTRDHM
#SBATCH -A psy53c17
#SBATCH --job-name=process_subject
#SBATCH --output=jobs/out%a_%A.out
#SBATCH --error=jobs/err%a_%A.err
#SBATCH --time=01:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe
#SBATCH --array=1-100  # Adjust the array size based on the number of subjects

source /data/users2/washbee/anaconda3latest/bin/activate wacvanalysis

# File containing the list of subject IDs
#SUBJ_FILE="test_subs.txt"
SUBJ_FILE="remaining.txt"
# Get the subject ID for this task
subj_id=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJ_FILE)

# Paths
csv_file_path="fastsurfer_measures.csv"
framework_name="FastSurfer"
gt_subject_base_path="/data/users2/washbee/speedrun/cortexode-data-rp/test"
fs_subject_base_path="/data/users2/washbee/fastsurfer-output/test"

# Run the Python script
python analyze_subject.py $subj_id $csv_file_path $framework_name $gt_subject_base_path $fs_subject_base_path
