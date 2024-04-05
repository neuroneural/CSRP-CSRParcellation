#!/bin/bash
#. /opt/miniconda3/bin/activate cortexode
. /opt/miniconda3/bin/activate csrf
cd /cortexode

# Parameters setup
declare -a versions=("1" "2")
declare -a gnns=("gcn" "gat")
declare -a gnn_layers=(3 4 5)
declare -a gat_heads=(1 2 3)

# Assuming $1 is provided as the equivalent of a job index
job_id=$1

# Initialize variables with default values
version=""
gnn=""
gnn_layer=""
gat_head=8  # Default value for gat_heads

# Calculate the number of jobs for "gcn" (not using gat_heads) and "gat" (using gat_heads)
gcn_jobs=$((${#versions[@]} * 1 * ${#gnn_layers[@]}))
gat_jobs=$((${#versions[@]} * 1 * ${#gnn_layers[@]} * ${#gat_heads[@]}))

total_jobs=$((gcn_jobs + gat_jobs))

if [ $job_id -lt $gcn_jobs ]; then
  # Handling gcn jobs
  version_index=$((job_id / ${#gnn_layers[@]}))
  gnn_layer_index=$((job_id % ${#gnn_layers[@]}))
  version=${versions[$version_index]}
  gnn="gcn"
  gnn_layer=${gnn_layers[$gnn_layer_index]}
else
  # Adjust index for gat jobs
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

python train.py  --version $version --gnn $gnn --gnn_layers $gnn_layer --gat_heads $gat_head --train_type='surf' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='/cortexode/ckpts/exp_csrf_gnn_1/model/' --init_dir='/cortexode/ckpts/exp_csrf_gnn_1/init/' --data_name='adni'  --surf_hemi='rh' --surf_type='wm' --n_epochs=100 --n_samples=150000 --tag='csrf' --solver='rk4' --step_size=0.1 --device='gpu' 

#python train.py --train_type='surf' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='/cortexode/ckpts/exp_csrf_v2/model/' --init_dir='/cortexode/ckpts/exp_csrf_v2/init/' --data_name='adni'  --surf_hemi='rh' --surf_type='wm' --n_epochs=401 --n_samples=150000 --tag='csrf' --solver='rk4' --step_size=0.1 --device='gpu'
#python train.py --train_type='surf' --data_dir='/speedrun/csrf-data-dev/' --model_dir='/cortexode/ckpts/exp_csrf_v3/model/' --init_dir='/cortexode/ckpts/exp_csrf_v3/init/' --data_name='adni'  --surf_hemi='rh' --surf_type='wm' --n_epochs=1 --n_samples=150000 --tag='csrf' --solver='rk4' --step_size=0.1 --device='gpu'
