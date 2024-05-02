#!/bin/bash
#. /opt/miniconda3/bin/activate cortexode
. /opt/miniconda3/bin/activate csrf

cd /cortexode

# Parameters setup
declare -a versions=("2")
declare -a gnns=("gat")  # Assuming "gcn" might not be included
declare -a gnn_layers=(2 3 4)
declare -a gat_heads=(1)


# Assuming $1 is provided as the equivalent of a job index
job_id=$1

# Initialize variables with default values
version=""
gnn=""
gnn_layer=""
gat_head=8  # Default value for gat_heads, might be adjusted based on job configuration

# Check if GNN types include "gcn" or "gat"
gcn_included=$(printf '%s\n' "${gnns[@]}" | grep -w "gcn" -c)
gat_included=$(printf '%s\n' "${gnns[@]}" | grep -w "gat" -c)

# Calculate the number of jobs
gcn_jobs=0
gat_jobs=0
if [ "$gcn_included" -eq "1" ]; then
    gcn_jobs=$((${#versions[@]} * ${#gnn_layers[@]}))
fi
if [ "$gat_included" -eq "1" ]; then
    gat_jobs=$((${#versions[@]} * ${#gnn_layers[@]} * ${#gat_heads[@]}))
fi

total_jobs=$((gcn_jobs + gat_jobs))

if [ "$job_id" -ge "$total_jobs" ]; then
    echo "Error: job_id ($job_id) exceeds total_jobs ($total_jobs)"
    exit 1
fi

# Determine if it's a "gcn" or "gat" job based on whether "gcn" is included
if [ "$job_id" -lt "$gcn_jobs" ]; then
    # Handling "gcn" jobs
    version_index=$((job_id / ${#gnn_layers[@]}))
    gnn_layer_index=$((job_id % ${#gnn_layers[@]}))
    version=${versions[$version_index]}
    gnn="gcn"
    gnn_layer=${gnn_layers[$gnn_layer_index]}
else
    # Adjust index for "gat" jobs
    adjusted_job_id=$((job_id - gcn_jobs))
    version_index=$((adjusted_job_id / (${#gnn_layers[@]} * ${#gat_heads[@]})))
    temp_index=$((adjusted_job_id % (${#gnn_layers[@]} * ${#gat_heads[@]})))
    gnn_layer_index=$((temp_index / ${#gat_heads[@]}))
    gat_head_index=$((temp_index % ${#gat_heads[@]}))
    version=${versions[$version_index]}
    gnn="gat"
    gnn_layer=${gnn_layers[$gnn_layer_index]}
    gat_head=${gat_heads[$gat_head_index]}
fi

# Execute Python script with parameters set above
echo "Running configuration: Version $version, GNN $gnn, GNN Layers $gnn_layer, GAT Heads $gat_head"
#python train.py  --version $version --gnn $gnn --gnn_layers $gnn_layer --gat_heads $gat_head --train_type='surf' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='/cortexode/ckpts/exp_csrf_gnn_1/model/' --init_dir='/cortexode/ckpts/exp_csrf_gnn_1/init/' --data_name='adni'  --surf_hemi='lh' --surf_type='wm' --n_epochs=100 --n_samples=150000 --tag='csrf' --solver='rk4' --step_size=0.1 --device='gpu' 
#python train.py --model_file 'model_wm_adni_lh_csrf_v1_gnngat_layers5_sf0.1_heads1_29epochs.pt' --version $version --gnn $gnn --gnn_layers $gnn_layer --gat_heads $gat_head --train_type='surf' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='/cortexode/ckpts/exp_csrf_gnn_2/model/' --init_dir='/cortexode/ckpts/exp_csrf_gnn_2/init/' --data_name='adni'  --surf_hemi='lh' --surf_type='wm' --n_epochs=60 --start_epoch=31 --n_samples=150000 --tag='csrf' --solver='rk4' --step_size=0.1 --device='gpu' 
python train.py --version $version --gnn $gnn --gnn_layers $gnn_layer --gat_heads $gat_head --train_type='surf' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='/cortexode/ckpts/exp_csrf_gnn_2/model/' --init_dir='/cortexode/ckpts/exp_csrf_gnn_2/init/' --data_name='adni'  --surf_hemi='lh' --surf_type='wm' --n_epochs=401 --start_epoch=1 --n_samples=150000 --tag='csrf' --solver='euler' --step_size=0.1 --device='gpu' 
