#!/bin/bash
#SBATCH -p qTRDHM
#SBATCH -A psy53c17
#SBATCH --job-name=fastav2.3
#SBATCH --output=jobs/out%A_%a.out
#SBATCH --error=jobs/err%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe

#start array at 1
source /data/users2/washbee/anaconda3_refresh/bin/activate isbianalysis

# File containing the list of subject IDs
#SUBJ_FILE="test_subs.txt"
SUBJ_FILE="../../test_subs.txt"
# Get the subject ID for this task
subj_id=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJ_FILE)
#start at 1 
# Paths
csv_file_path="../csvs/freesurfercsrp_measures_v2.3.csv"
framework_name="freesurfercsrp"
gt_subject_base_path="/data/users2/washbee/speedrun/cortexode-data-rp/test"
base_path="/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv4_0/result/"

# Run the Python script
python ../../../analyze_subject_v2.3.py --subj_id=$subj_id --csv_file_path=$csv_file_path --framework_name=$framework_name --base_path=$base_path --condition="d" --gt_subject_base_path=$gt_subject_base_path
