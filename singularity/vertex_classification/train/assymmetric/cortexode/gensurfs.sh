#!/bin/bash

# Activate the conda environment
. /opt/miniconda3/bin/activate csrf

# Change to the cortexode directory
cd /cortexode

# Hardcoded model paths
SEG_MODEL_SOURCE_PATH="/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_csrf_gnn_0/model/model_seg_hcp_Unet.pt"

# Parameters
STEP_SIZE=0.1
SOLVER="euler"

# Version 0
MODEL_DIR_V0=/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_csrf_gnn_0/model/
DEF_MODEL_PATH_GM_LH_V0="model_gm_hcp_lh_csrf_v0_gnnbaseline_sf0.1_200epochs_euler.pt"
DEF_MODEL_PATH_GM_RH_V0="model_gm_hcp_rh_csrf_v0_gnnbaseline_sf0.1_160epochs_euler.pt"
DEF_MODEL_PATH_WM_LH_V0="model_wm_hcp_lh_csrf_v0_gnnbaseline_sf0.1_200epochs_euler.pt"
DEF_MODEL_PATH_WM_RH_V0="model_wm_hcp_rh_csrf_v0_gnnbaseline_sf0.1_170epochs_euler.pt"

parc_init_dir_VERSION_0="/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_vc_assym_gnn_0/parc_init"

# Run the Python scripts with the specified parameters
python generate_parcellation_surfaces.py --gnn_layers=0 --model_type=baseline --gnn baseline --data_dir='/speedrun/cortexode-data-rp/' --data_name='hcp' --n_samples=150000 --tag='csrf' --device='gpu' --model_dir=$MODEL_DIR_V0 --segmentation_model_path $SEG_MODEL_SOURCE_PATH --model_file $DEF_MODEL_PATH_GM_LH_V0 --version 0 --surf_hemi lh --surf_type gm --step_size $STEP_SIZE --solver $SOLVER --parc_init_dir $parc_init_dir_VERSION_0 &
pid1=$!  # Store the process ID of the first process

python generate_parcellation_surfaces.py --gnn_layers=0 --model_type=baseline --gnn baseline --data_dir='/speedrun/cortexode-data-rp/' --data_name='hcp' --n_samples=150000 --tag='csrf' --device='gpu' --model_dir=$MODEL_DIR_V0 --segmentation_model_path $SEG_MODEL_SOURCE_PATH --model_file $DEF_MODEL_PATH_GM_RH_V0 --version 0 --surf_hemi rh --surf_type gm --step_size $STEP_SIZE --solver $SOLVER --parc_init_dir $parc_init_dir_VERSION_0 &
pid2=$!

wait $pid1
wait $pid2