#!/bin/bash
. /opt/miniconda3/bin/activate csrf

cd /cortexode

# Set the variables
test_type="pred"
data_dir='/speedrun/cortexode-data-rp/'
model_dir='/cortexode/ckpts/isbi_gnn_1/model/'
init_dir='/cortexode/ckpts/isbi_gnn_1/init/'
result_dir="/cortexode/ckpts/isbi_gnn_1/result/"  # Set the directory for result surfaces
data_name="hcp"  # Dataset name (e.g., 'hcp')
surf_hemi="lh"  # Hemisphere ('lh' for left, 'rh' for right)
seg_model_file="model_seg_hcp_Unet_200epochs.pt"  # Segmentation model file
wm_model_file="model_wm_hcp_lh_vc_v2_csrvc_layers10_sf0.1_heads1_10epochs_euler.pt"  # White matter model file
gm_model_file="model_gm_hcp_lh_vc_v2_csrvc_layers10_sf0.1_heads1_10epochs_euler.pt"  # Gray matter model file

# Execute the Python script with the set parameters
python generateIsbiTestSurfaces.py \
  --test_type "$test_type" \
  --data_dir "$data_dir" \
  --model_dir "$model_dir" \
  --init_dir "$init_dir" \
  --result_dir "$result_dir" \
  --data_name "$data_name" \
  --surf_hemi "$surf_hemi" \
  --device "cuda" \
  --solver "euler" \
  --seg_model_file "$seg_model_file" \
  --wm_model_file "$wm_model_file" \
  --gm_model_file "$gm_model_file" \
  --gnn 'gat' \
  --gnn_layers 10 \
  --gat_heads 1 \
    