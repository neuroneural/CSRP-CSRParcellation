#!/bin/bash

# Activate the conda environment
. /opt/miniconda3/bin/activate csrf

# Change to the cortexode directory
cd /cortexode

# Hardcoded model paths
SEG_MODEL_SOURCE_PATH="/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_csrf_gnn_0/model/model_seg_hcp_Unet.pt"
SEG_MODEL_DEST_PATH="./wacvanalysis/frameworks/csrf/model/model_seg_hcp_Unet.pt"

# Check if the segmentation model exists in the destination directory, if not, copy it there
if [ ! -f "$SEG_MODEL_DEST_PATH" ]; then
    echo "Copying $SEG_MODEL_SOURCE_PATH to $SEG_MODEL_DEST_PATH for reproducibility..."
    cp "$SEG_MODEL_SOURCE_PATH" "$SEG_MODEL_DEST_PATH"
else
    echo "$SEG_MODEL_DEST_PATH already exists."
fi

# Parameters
STEP_SIZE=0.1
SOLVER="euler"

# Version 0
MODEL_DIR_V0=/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_csrf_gnn_2/model/
DEF_MODEL_PATH_GM_LH_V0="model_gm_hcp_lh_csrf_v3_gnngat_layers3_sf0.1_heads1_180epochs_euler.pt"
DEF_MODEL_PATH_GM_RH_V0="model_gm_hcp_rh_csrf_v3_gnngat_layers3_sf0.1_heads1_160epochs_euler.pt"
DEF_MODEL_PATH_WM_LH_V0="model_wm_hcp_lh_csrf_v3_gnngat_layers4_sf0.1_heads1_200epochs_euler.pt"
DEF_MODEL_PATH_WM_RH_V0="model_wm_hcp_rh_csrf_v3_gnngat_layers3_sf0.1_heads1_200epochs_euler.pt"

RESULT_DIR_VERSION_0="./wacvanalysis/frameworks/csrf/result"

# Run the Python scripts with the specified parameters
python ./wacvanalysis/frameworks/generate_testsurfaces.py --gnn_layers 3 --gat_heads 1 --gnn gat --data_dir='/speedrun/cortexode-data-rp/' --data_name='hcp' --n_samples=150000 --tag='csrf' --device='gpu' --model_dir=$MODEL_DIR_V0 --segmentation_model_path $SEG_MODEL_DEST_PATH --model_file $DEF_MODEL_PATH_GM_LH_V0 --version 3 --surf_hemi lh --surf_type gm --step_size $STEP_SIZE --solver $SOLVER --result_dir $RESULT_DIR_VERSION_0 & 
python ./wacvanalysis/frameworks/generate_testsurfaces.py --gnn_layers 3 --gat_heads 1 --gnn gat --data_dir='/speedrun/cortexode-data-rp/' --data_name='hcp' --n_samples=150000 --tag='csrf' --device='gpu' --model_dir=$MODEL_DIR_V0 --segmentation_model_path $SEG_MODEL_DEST_PATH --model_file $DEF_MODEL_PATH_GM_RH_V0 --version 3 --surf_hemi rh --surf_type gm --step_size $STEP_SIZE --solver $SOLVER --result_dir $RESULT_DIR_VERSION_0 &
wait 
python ./wacvanalysis/frameworks/generate_testsurfaces.py --gnn_layers 4 --gat_heads 1 --gnn gat --data_dir='/speedrun/cortexode-data-rp/' --data_name='hcp' --n_samples=150000 --tag='csrf' --device='gpu' --model_dir=$MODEL_DIR_V0 --segmentation_model_path $SEG_MODEL_DEST_PATH --model_file $DEF_MODEL_PATH_WM_LH_V0 --version 3 --surf_hemi lh --surf_type wm --step_size $STEP_SIZE --solver $SOLVER --result_dir $RESULT_DIR_VERSION_0 &
python ./wacvanalysis/frameworks/generate_testsurfaces.py --gnn_layers 3 --gat_heads 1 --gnn gat --data_dir='/speedrun/cortexode-data-rp/' --data_name='hcp' --n_samples=150000 --tag='csrf' --device='gpu' --model_dir=$MODEL_DIR_V0 --segmentation_model_path $SEG_MODEL_DEST_PATH --model_file $DEF_MODEL_PATH_WM_RH_V0 --version 3 --surf_hemi rh --surf_type wm --step_size $STEP_SIZE --solver $SOLVER --result_dir $RESULT_DIR_VERSION_0 &
wait
