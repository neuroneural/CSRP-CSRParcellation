#!/bin/bash
. /opt/miniconda3/bin/activate csrf
cd /cortexode

# Parameters setup
version=3
gnn="gat"
gat_head=1
train_type="surf"
data_dir="/speedrun/cortexode-data-rp/"
model_dir="/cortexode/ckpts/hcp_csrf_gnn_2/model/"
init_dir="/cortexode/ckpts/hcp_csrf_gnn_2/init/"
data_name="hcp"
n_epochs=201
n_samples=150000
tag="csrf"
solver="euler"
step_size=0.1
device="gpu"

# Get the GNN layer based on the 1
gnn_layers=(2 3 4)
gnn_layer=${gnn_layers[$1]}

# Determine model file and start epoch based on gnn_layer
case $gnn_layer in
    2)
        model_file="model_gm_hcp_rh_csrf_v3_gnngat_layers2_sf0.1_heads1_200epochs_euler.pt"
        start_epoch=201
        ;;
    3)
        model_file="model_gm_hcp_rh_csrf_v3_gnngat_layers3_sf0.1_heads1_170epochs_euler.pt"
        start_epoch=171
        ;;
    4)
        model_file="model_gm_hcp_rh_csrf_v3_gnngat_layers4_sf0.1_heads1_140epochs_euler.pt"
        start_epoch=141
        ;;
esac

# Execute Python script with parameters set above
echo "Running configuration: Version $version, GNN $gnn, GNN Layers $gnn_layer, GAT Heads $gat_head"
python train_ablation.py --model_file "$model_file" --version "$version" --gnn "$gnn" --gnn_layers "$gnn_layer" --gat_heads "$gat_head" --train_type="$train_type" --data_dir="$data_dir" --model_dir="$model_dir" --init_dir="$init_dir" --data_name="$data_name" --surf_hemi="rh" --surf_type="gm" --n_epochs="$n_epochs" --start_epoch="$start_epoch" --n_samples="$n_samples" --tag="$tag" --solver="$solver" --step_size="$step_size" --device="$device"
