#!/bin/bash

# This script is executed inside the Singularity container.

# Retrieve the arguments passed from the Slurm script
surf_type="$1"
data_name="$2"
hemisphere="$3"
version="$4"
model_type="$5"
layers="$6"
heads="$7"
epoch="$8"
solver="$9"
reconstruction="${10}"
classification="${11}"
random_number="${12}"
MODEL_FILE="${13}"
MODEL_DIR="${14}"

echo "Inside container script:"
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

# Activate the Conda environment
source /opt/miniconda3/bin/activate csrf

# Navigate to the appropriate directory if necessary
cd /cortexode/

# Run the Python script with the provided arguments
python generateISBITestSurfaces.py \
    --data_dir "/speedrun/cortexode-data-rp/" \
    --surf_type "$surf_type" \
    --data_name "$data_name" \
    --surf_hemi "$hemisphere" \
    --version "$version" \
    --model_type "$model_type" \
    --gnn_layers "$layers" \
    --gnn 'gat' \
    --gat_heads "$heads" \
    --start_epoch "$epoch" \
    --solver "$solver" \
    --recon "$reconstruction" \
    --classification "$classification" \
    --random_number "$random_number" \
    --seg_model_file "model_seg_hcp_Unet_200epochs.pt" \
    --model_file_gm "$MODEL_FILE" \
    --init_dir "/cortexode/ckpts/isbi/isbi_gnnv3undirectedjoint_0/init" \
    --result_dir "/cortexode/ckpts/isbi/isbi_gnnv3undirectedjoint_0/result/" \
    --model_dir "$MODEL_DIR"
