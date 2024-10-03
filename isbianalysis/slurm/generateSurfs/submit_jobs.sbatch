#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=20g
#SBATCH -p qTRDGPU
#SBATCH --gres=gpu:RTX:1
#SBATCH -t 5-00:00
#SBATCH -J gensurfis
#SBATCH -e jobs/error%A_%a.err
#SBATCH -o jobs/out%A_%a.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe
#SBATCH --exclude=arctrddgxa001
#SBATCH --array=1-M  # Replace M with the total number of lines (models) in your new CSV

# Brief pause before starting
sleep 5s

# Load Singularity module
module load singularity/3.10.2

# Define the path to your new CSV file
CSV_FILE="/data/users2/washbee/CortexODE-CSRFusionNet/isbianalysis/models_to_run.csv"

# Check if the CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    echo "CSV file not found at $CSV_FILE. Exiting."
    exit 1
fi

# Get the header line count (should be 1)
HEADER_LINES=1

# Calculate the actual line number in the CSV file
TASK_LINE=$((SLURM_ARRAY_TASK_ID + HEADER_LINES))

# Extract the specific line based on SLURM_ARRAY_TASK_ID
LINE=$(sed -n "${TASK_LINE}p" "$CSV_FILE")

# Check if the LINE variable is not empty
if [ -z "$LINE" ]; then
    echo "No data found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}. Exiting."
    exit 1
fi

# Read the CSV line into variables
IFS=',' read -r surf_type data_name hemisphere version model_type layers heads epoch solver reconstruction classification random_number MODEL_FILE MODEL_DIR <<< "$LINE"

# Debugging: Print out the variables to verify correct assignment
echo "surf_type: $surf_type"
echo "data_name: $data_name"
echo "hemisphere: $hemisphere"
echo "version: $version"
echo "model_type: $model_type"
echo "layers: $layers"
echo "heads: $heads"
echo "epoch: $epoch"
echo "solver: $solver"
echo "reconstruction: $reconstruction"
echo "classification: $classification"
echo "random_number: $random_number"
echo "MODEL_FILE: $MODEL_FILE"
echo "MODEL_DIR: $MODEL_DIR"

# Trim whitespace from variables (optional but recommended)
surf_type=$(echo "$surf_type" | xargs)
data_name=$(echo "$data_name" | xargs)
hemisphere=$(echo "$hemisphere" | xargs)
version=$(echo "$version" | xargs)
model_type=$(echo "$model_type" | xargs)
layers=$(echo "$layers" | xargs)
heads=$(echo "$heads" | xargs)
epoch=$(echo "$epoch" | xargs)
solver=$(echo "$solver" | xargs)
reconstruction=$(echo "$reconstruction" | xargs)
classification=$(echo "$classification" | xargs)
random_number=$(echo "$random_number" | xargs)
MODEL_FILE=$(echo "$MODEL_FILE" | xargs)
MODEL_DIR=$(echo "$MODEL_DIR" | xargs)

# Path to your shell script that activates the Conda environment and runs the Python script
INNER_SHELL_SCRIPT="/cortexode/isbianalysis/slurm/generateSurfs/run_generate_surfacev2.sh"

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
        "$reconstruction" \
        "$classification" \
        "$random_number" \
        "$MODEL_FILE" \
        "$MODEL_DIR"

# Wait for the singularity command to finish
wait

# Brief pause after completion
sleep 5s
