#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=20g
#SBATCH -p qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH -t 5-00:00
#SBATCH -J gensurfis
#SBATCH -e jobs/error%A_%a.err
#SBATCH -o jobs/out%A_%a.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe
#SBATCH --exclude=arctrddgxa001
#SBATCH --array=1-N  # Replace N with the total number of lines (models) in your CSV

# Enable debugging (optional)
# set -x

# Brief pause before starting
sleep 5s

# Load Singularity module
module load singularity/3.10.2

# Define the path to your CSV file
CSV_FILE="/data/users2/washbee/CortexODE-CSRFusionNet/isbianalysis/top_5_models_per_group.csv"

# Check if the CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    echo "CSV file not found at $CSV_FILE. Exiting."
    exit 1
fi

# Extract the specific line based on SLURM_ARRAY_TASK_ID
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CSV_FILE")

# Check if the LINE variable is not empty
if [ -z "$LINE" ]; then
    echo "No data found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}. Exiting."
    exit 1
fi

# Read the CSV line into variables
IFS=',' read -r file_name surf_type hemisphere layers solver heads version random_number model_type classification reconstruction epoch recon_error dice_error <<< "$LINE"

# Debugging: Print out the variables to verify correct assignment
echo "file_name: $file_name"
echo "surf_type: $surf_type"
echo "hemisphere: $hemisphere"
echo "layers: $layers"
echo "solver: $solver"
echo "heads: $heads"
echo "version: $version"
echo "random_number: $random_number"
echo "model_type: $model_type"
echo "classification: $classification"
echo "reconstruction: $reconstruction"
echo "epoch: $epoch"
echo "recon_error: $recon_error"
echo "dice_error: $dice_error"

# Trim whitespace from variables (optional but recommended)
surf_type=$(echo "$surf_type" | xargs)
hemisphere=$(echo "$hemisphere" | xargs)
layers=$(echo "$layers" | xargs)
solver=$(echo "$solver" | xargs)
heads=$(echo "$heads" | xargs)
version=$(echo "$version" | xargs)
random_number=$(echo "$random_number" | xargs)
model_type=$(echo "$model_type" | xargs)
classification=$(echo "$classification" | xargs)
reconstruction=$(echo "$reconstruction" | xargs)
epoch=$(echo "$epoch" | xargs)
recon_error=$(echo "$recon_error" | xargs)
dice_error=$(echo "$dice_error" | xargs)

# Extract data_name from file_name
data_name=$(echo "$file_name" | awk -F'_' '{print $3}')
echo "data_name (extracted): $data_name"

# Convert random_number from float to int by removing the last two characters
random_number_int="${random_number::-2}"
echo "random_number_int: $random_number_int"

# Correct Mapping of Reconstruction Flag
if [[ "$reconstruction" =~ ^[Tt]([Rr]ue)?$ ]]; then
    reconstruction_bool="recon"
else
    reconstruction_bool="norecon"
fi

# Classification Flag Mapping remains correct
if [[ "$classification" =~ ^[Tt]([Rr]ue)?$ ]]; then
    classification_bool="class"
else
    classification_bool="noclass"
fi

echo "reconstruction_bool: $reconstruction_bool"
echo "classification_bool: $classification_bool"

# Construct the model filename based on the required pattern
MODEL_FILE="model_${surf_type}_${data_name}_${hemisphere}_vc_${version}_${model_type}_layers${layers}_sf0.1_heads${heads}_${epoch}epochs_${solver}_${reconstruction_bool}_${classification_bool}_${random_number_int}.pt"
MODEL_DIR="/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv3undirectedjoint_0/model/"
echo "Constructed MODEL_FILE_PATH: $MODEL_FILE_PATH"

# Define the save_basename by removing the .pt extension
SAVE_BASENAME="${MODEL_FILE%.pt}"
echo "SAVE_BASENAME: $SAVE_BASENAME"

# Check if the model file exists
if [ ! -f "${MODEL_DIR}${MODEL_FILE}" ]; then
    echo "Model file not found at ${MODEL_DIR}${MODEL_FILE}. Skipping."
    exit 1
fi

# Path to your shell script that activates the Conda environment and runs the Python script
INNER_SHELL_SCRIPT="/cortexode/isbianalysis/slurm/run_generate_surface.sh"

# Execute the shell script inside the Singularity container
singularity exec --nv \
    --bind /data,\
/data/users2/washbee/speedrun/:/speedrun,\
/data/users2/washbee/CortexODE-CSRFusionNet:/cortexode \
    /data/users2/washbee/containers/csrf_sandbox/ \
    bash "$INNER_SHELL_SCRIPT" \
        "$surf_type" \
        "$data_name" \
        "$hemisphere" \
        "$version" \
        "$model_type" \
        "$layers" \
        "$heads" \
        "$epoch" \
        "$solver" \
        "$reconstruction_bool" \
        "$classification_bool" \
        "$random_number_int" \
        "$MODEL_FILE" \
        "$MODEL_DIR"

# Wait for the singularity command to finish
wait

# Brief pause after completion
sleep 5s
