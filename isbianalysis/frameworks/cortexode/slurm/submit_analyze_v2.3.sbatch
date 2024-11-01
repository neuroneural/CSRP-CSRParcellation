#!/bin/bash
#SBATCH -p qTRDHM
#SBATCH -A psy53c17
#SBATCH --job-name=alyzsub_csrp
#SBATCH --output=jobs/out%A_%a.out
#SBATCH --error=jobs/err%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe

# Activate the Anaconda environment
condition='c'
echo 'condition ' $condition
source /data/users2/washbee/anaconda3_refresh/bin/activate isbianalysis
cd /data/users2/washbee/CortexODE-CSRFusionNet/isbianalysis/frameworks/cortexode/slurm/
# File containing the list of subject IDs
SUBJ_FILE="../../test_subs.txt"

# Get the subject ID for this task
subj_id=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJ_FILE)

echo $subj_id
# Define Paths
csv_file_path="../csvs/cortexode_dice_${condition}.csv"
framework_name="cortexode"


# Updated paths where both predictions and ground truths are stored
# Assuming that the ground truths are now within the same directories as predictions
 # For example: /data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv3undirectedjoint_0/result/a/"#gm/lh/
base_path="/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv4_0/result/"
gt_subject_base_path="/data/users2/washbee/speedrun/cortexode-data-rp/test"
# gt_subject_base_path="/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv3undirectedjoint_0/result/400_100_100/a"
# proj_subject_base_path="/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv3undirectedjoint_0/result/400_100_100/a"

# gt_subject_base_path="/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv3undirectedjoint_0/result/400_100_100/b"
# proj_subject_base_path="/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv3undirectedjoint_0/result/400_100_100/b"

# Run the Python script
echo subj_id $subj_id 
echo csv_file_path $csv_file_path 
echo framework_name $framework_name 
echo base_path $base_path 
echo condition $condition
python ../../../analyze_subject_v2.3.py --subj_id=$subj_id --csv_file_path=$csv_file_path --framework_name=$framework_name --base_path=$base_path --condition=$condition --gt_subject_base_path=$gt_subject_base_path
