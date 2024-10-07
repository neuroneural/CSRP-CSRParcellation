#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=20g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:A100:1
#SBATCH -t 1-00:00
#SBATCH -J gensurfis
#SBATCH -e jobs/error%A_%a.err
#SBATCH -o jobs/out%A_%a.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe
#SBATCH --exclude=arctrddgxa001,arctrdagn004
#SBATCH --array=1-M  # Replace M with the total number of lines (excluding the header) in your new CSV

# Brief pause before starting
sleep 5s

# Load Singularity module
module load singularity

# Define the path to your new CSV file

# Get the header line count (should be 1)

# Calculate the actual line number in the CSV file
TASK_LINE=${SLURM_ARRAY_TASK_ID}

wm_model_dir=/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_csrf_gnn_0/model
gm_model_dir=/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_csrf_gnn_0/model

case='c'
data_name='hcp'
surf_hemi=('lh' 'rh')
surf_hemi=${surf_hemi[${TASK_LINE} - 1]}
solver='euler'
gnn_layers=0
gat_heads=0
model_file_wm=('model_wm_hcp_lh_csrf_v0_gnnbaseline_sf0.1_90epochs_euler.pt' 'model_wm_hcp_rh_csrf_v0_gnnbaseline_sf0.1_90epochs_euler.pt')
model_file_wm=${model_file_wm[$TASK_LINE - 1]}
wm_model_file_combined="no"
wm_model_file_deformation="no"
wm_model_file_classification='no'
gm_model_file_combined="no"
model_file_gm=('model_gm_hcp_lh_csrf_v0_gnnbaseline_sf0.1_90epochs_euler.pt' 'model_gm_hcp_rh_csrf_v0_gnnbaseline_sf0.1_90epochs_euler.pt')
model_file_gm=${model_file_gm[$TASK_LINE - 1]}
gm_model_file_classification='no'

# Debugging: Print out the variables to verify correct assignment
echo "data_name: $data_name"
echo "surf_hemi: $surf_hemi"
echo "solver: $solver"
echo "gat_heads: $gat_heads"
echo "wm_model_dir: $wm_model_dir"
echo "wm_model_file_combined: $wm_model_file_combined"
echo "model_file_wm: $model_file_wm"
echo "wm_model_file_classification: $wm_model_file_classification"
echo "gm_model_dir: $gm_model_dir"
echo "gm_model_file_combined: $gm_model_file_combined"
echo "model_file_gm: $model_file_gm"
echo "gm_model_file_classification: $gm_model_file_classification"

# Trim whitespace from variables (optional but recommended)
data_name=$(echo "$data_name" | xargs)
surf_hemi=$(echo "$surf_hemi" | xargs)
gnn_layers=$(echo "$gnn_layers" | xargs)
solver=$(echo "$solver" | xargs)
gat_heads=$(echo "$gat_heads" | xargs)
wm_model_dir=$(echo "$wm_model_dir" | xargs)
wm_model_file_combined=$(echo "$wm_model_file_combined" | xargs)
wm_model_file_deformation=$(echo "$wm_model_file_deformation" | xargs)
wm_model_file_classification=$(echo "$wm_model_file_classification" | xargs)
gm_model_dir=$(echo "$gm_model_dir" | xargs)
gm_model_file_combined=$(echo "$gm_model_file_combined" | xargs)
gm_model_file_deformation=$(echo "$gm_model_file_deformation" | xargs)
gm_model_file_classification=$(echo "$gm_model_file_classification" | xargs)

# Path to your shell script that activates the Conda environment and runs the Python script
INNER_SHELL_SCRIPT="/cortexode/isbianalysis/slurm/generateSurfs/run_generate_surfacev2.sh"

# Execute the shell script inside the Singularity container
singularity exec --nv \
    --bind /data,/data/users2/washbee/speedrun/:/speedrun,/data/users2/washbee/CortexODE-CSRFusionNet:/cortexode \
    /data/users2/washbee/containers/csrf_sandbox/ \
    bash "$INNER_SHELL_SCRIPT" \
        "$case" \
        "$data_name" \
        "$surf_hemi" \
        "$gnn_layers" \
        "$solver" \
        "$gat_heads" \
        "$wm_model_dir" \
        "$wm_model_file_combined" \
        "$model_file_wm" \
        "$wm_model_file_classification" \
        "$gm_model_dir" \
        "$gm_model_file_combined" \
        "$model_file_gm" \
        "$gm_model_file_classification"

# Wait for the singularity command to finish
wait

# Brief pause after completion
sleep 5s
