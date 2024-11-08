#!/bin/bash
. /opt/miniconda3/bin/activate csrf
cd /cortexode

# Parameters setup
declare -a versions=("3")
declare -a gnns=("gat")  # Assuming "gcn" might not be included
declare -a gnn_layers=(2 4 6 8 10 12)
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
python train_CSRVertex_labeling.py --data_dir='/speedrun/cortexode-data-rp/' --model_dir='/cortexode/ckpts/isbi/isbi_gnnv3undirectedseperatedice_0/model/' --result_dir='/cortexode/ckpts/isbi/isbi_gnnv3undirectedseperatedice_0/result/' --model_type 'csrvc' --patience 'standard' --visualize 'yes' --version $version --gnn $gnn --gnn_layers $gnn_layer --gat_heads $gat_head --train_type='surfvc' --data_name='hcp'  --surf_hemi='lh' --surf_type='gm' --n_epochs=401 --start_epoch=1 --n_samples=150000 --tag='vc' --device='gpu' 
