#!/bin/bash
#SBATCH -p qTRDHM
#SBATCH -A psy53c17
#SBATCH --job-name=alyzsub_csrp
#SBATCH --output=jobs/out%a_%A.out
#SBATCH --error=jobs/err%a_%A.err
#SBATCH --time=10:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe

# Activate the Anaconda environment
source /data/users2/washbee/anaconda3latest/bin/activate wacvanalysis

# File containing the list of subject IDs
SUBJ_FILE="../../test_subs.txt"

# Get the subject ID for this task
subj_id=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJ_FILE)

# Define Paths
csv_file_path="../csvs/csrp_measures.csv"
framework_name="csrp"

# Updated paths where both predictions and ground truths are stored
# Assuming that the ground truths are now within the same directories as predictions
# For example: /data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv3undirectedjoint_0/result/a/"#gm/lh/
gt_subject_base_path="/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv3undirectedjoint_0/result/a/"
proj_subject_base_path="/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv3undirectedjoint_0/result/a/"

# Run the Python script
echo $subj_id $csv_file_path $framework_name $gt_subject_base_path $proj_subject_base_path
python ../../../analyze_subject_csrp.py \
    $subj_id \
    $csv_file_path \
    $framework_name \
    $gt_subject_base_path \
    $proj_subject_base_path
