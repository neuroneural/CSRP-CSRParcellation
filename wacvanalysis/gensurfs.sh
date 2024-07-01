#!/bin/bash
. /opt/miniconda3/bin/activate csrf
cd /cortexode

# Hardcoded model paths
SEG_MODEL_PATH="/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_csrf_gnn_0/model/model_seg_hcp_Unet.pt"
RESULT_DIR_VERSION_0="/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_csrf_gnn_0/result"
RESULT_DIR_VERSION_3="/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_csrf_gnn_2/result"

# Parameters
STEP_SIZE=0.1
SOLVER="euler"

# Version 0
MODEL_DIR_V0=/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_csrf_gnn_0/model/
DEF_MODEL_PATH_GM_LH_V0="model_gm_hcp_lh_csrf_v0_gnnbaseline_sf0.1_200epochs_euler.pt"
DEF_MODEL_PATH_GM_RH_V0="model_gm_hcp_rh_csrf_v0_gnnbaseline_sf0.1_160epochs_euler.pt"
DEF_MODEL_PATH_WM_LH_V0="model_wm_hcp_lh_csrf_v0_gnnbaseline_sf0.1_200epochs_euler.pt"
DEF_MODEL_PATH_WM_RH_V0="model_wm_hcp_rh_csrf_v0_gnnbaseline_sf0.1_170epochs_euler.pt"

# Version 3
MODEL_DIR_V3=/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_csrf_gnn_2/model/
DEF_MODEL_PATH_GM_LH_V3="model_gm_hcp_lh_csrf_v3_gnngat_layers4_sf0.1_heads1_130epochs_euler.pt"
DEF_MODEL_PATH_GM_RH_V3="model_gm_hcp_rh_csrf_v3_gnngat_layers3_sf0.1_heads1_160epochs_euler.pt"
DEF_MODEL_PATH_WM_LH_V3="model_wm_hcp_lh_csrf_v3_gnngat_layers2_sf0.1_heads1_180epochs_euler.pt"
DEF_MODEL_PATH_WM_RH_V3="model_wm_hcp_rh_csrf_v3_gnngat_layers2_sf0.1_heads1_160epochs_euler.pt"

python generate_testsurfaces.py --gnn baseline --data_dir='/speedrun/cortexode-data-rp/' --data_name='hcp'  --n_samples=150000 --tag='csrf' --device='gpu' --model_dir=$MODEL_DIR_V0 --segmentation_model_path $SEG_MODEL_PATH --model_file $DEF_MODEL_PATH_GM_LH_V0 --version 0 --surf_hemi lh --surf_type gm --step_size $STEP_SIZE --solver $SOLVER --result_dir $RESULT_DIR_VERSION_0
python generate_testsurfaces.py --gnn baseline --data_dir='/speedrun/cortexode-data-rp/' --data_name='hcp'  --n_samples=150000 --tag='csrf' --device='gpu' --model_dir=$MODEL_DIR_V0 --segmentation_model_path $SEG_MODEL_PATH --model_file $DEF_MODEL_PATH_GM_RH_V0 --version 0 --surf_hemi rh --surf_type gm --step_size $STEP_SIZE --solver $SOLVER --result_dir $RESULT_DIR_VERSION_0
python generate_testsurfaces.py --gnn baseline --data_dir='/speedrun/cortexode-data-rp/' --data_name='hcp'  --n_samples=150000 --tag='csrf' --device='gpu' --model_dir=$MODEL_DIR_V0 --segmentation_model_path $SEG_MODEL_PATH --model_file $DEF_MODEL_PATH_WM_LH_V0 --version 0 --surf_hemi lh --surf_type wm --step_size $STEP_SIZE --solver $SOLVER --result_dir $RESULT_DIR_VERSION_0
python generate_testsurfaces.py --gnn baseline --data_dir='/speedrun/cortexode-data-rp/' --data_name='hcp'  --n_samples=150000 --tag='csrf' --device='gpu' --model_dir=$MODEL_DIR_V0 --segmentation_model_path $SEG_MODEL_PATH --model_file $DEF_MODEL_PATH_WM_RH_V0 --version 0 --surf_hemi rh --surf_type wm --step_size $STEP_SIZE --solver $SOLVER --result_dir $RESULT_DIR_VERSION_0

# Run evaluation for version 3 with specified number of layers
python generate_testsurfaces.py --gnn gat --gat_heads 1 --data_dir='/speedrun/cortexode-data-rp/' --data_name='hcp'  --n_samples=150000 --tag='csrf' --device='gpu' --model_dir=$MODEL_DIR_V3 --segmentation_model_path $SEG_MODEL_PATH --model_file $DEF_MODEL_PATH_GM_LH_V3 --version 3 --gnn_layers 4 --surf_hemi lh --surf_type gm --step_size $STEP_SIZE --solver $SOLVER --result_dir $RESULT_DIR_VERSION_3
python generate_testsurfaces.py --gnn gat --gat_heads 1 --data_dir='/speedrun/cortexode-data-rp/' --data_name='hcp'  --n_samples=150000 --tag='csrf' --device='gpu' --model_dir=$MODEL_DIR_V3 --segmentation_model_path $SEG_MODEL_PATH --model_file $DEF_MODEL_PATH_GM_RH_V3 --version 3 --gnn_layers 3 --surf_hemi rh --surf_type gm --step_size $STEP_SIZE --solver $SOLVER --result_dir $RESULT_DIR_VERSION_3
python generate_testsurfaces.py --gnn gat --gat_heads 1 --data_dir='/speedrun/cortexode-data-rp/' --data_name='hcp'  --n_samples=150000 --tag='csrf' --device='gpu' --model_dir=$MODEL_DIR_V3 --segmentation_model_path $SEG_MODEL_PATH --model_file $DEF_MODEL_PATH_WM_LH_V3 --version 3 --gnn_layers 2 --surf_hemi lh --surf_type wm --step_size $STEP_SIZE --solver $SOLVER --result_dir $RESULT_DIR_VERSION_3
python generate_testsurfaces.py --gnn gat --gat_heads 1 --data_dir='/speedrun/cortexode-data-rp/' --data_name='hcp'  --n_samples=150000 --tag='csrf' --device='gpu' --model_dir=$MODEL_DIR_V3 --segmentation_model_path $SEG_MODEL_PATH --model_file $DEF_MODEL_PATH_WM_RH_V3 --version 3 --gnn_layers 2 --surf_hemi rh --surf_type wm --step_size $STEP_SIZE --solver $SOLVER --result_dir $RESULT_DIR_VERSION_3

