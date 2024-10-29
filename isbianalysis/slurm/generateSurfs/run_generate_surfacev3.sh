#!/bin/bash

# This script is executed inside the Singularity container.

# Retrieve the arguments passed from the Slurm script
case="$1"
data_name="$2"
surf_hemi="$3"
gnn_layers="$4"
solver="$5"
gat_heads="$6"
wm_model_dir="$7"
wm_model_file_combined="$8"
model_file_wm_deformation="$9"
model_file_wm_classification="${10}"
gm_model_dir="${11}"
gm_model_file_combined="${12}"
model_file_gm_deformation="${13}"
model_file_gm_classification="${14}"

model_type="${15}"

echo "Inside container script:"
echo "case: $case"
echo "data_name: $data_name"
echo "surf_hemi: $surf_hemi"
echo "gnn_layers: $gnn_layers"
echo "solver: $solver"
echo "gat_heads: $gat_heads"
echo "wm_model_dir: $wm_model_dir"
echo "wm_model_file_combined: $wm_model_file_combined"
echo "model_file_wm_deformation: $model_file_wm_deformation"
echo "model_file_wm_classification: $model_file_wm_classification"
echo "gm_model_dir: $gm_model_dir"
echo "gm_model_file_combined: $gm_model_file_combined"
echo "model_file_gm_deformation: $model_file_gm_deformation"
echo "model_file_gm_classification: $model_file_gm_classification"
echo "model_type: $model_type"

# Activate the Conda environment
source /opt/miniconda3/bin/activate csrf

# Navigate to the appropriate directory if necessary
cd /cortexode/

# Build the Python command with the provided arguments
python_command="python generateISBITestSurfaces.py \
    --data_dir '/speedrun/cortexode-data-rp/' \
    --data_name '$data_name' \
    --surf_hemi '$surf_hemi' \
    --gnn_layers '$gnn_layers' \
    --gnn 'gat' \
    --gat_heads '$gat_heads' \
    --solver '$solver' \
    --result_dir '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv4_0/result/' \
    --result_subdir $case \
    --model_type '$model_type' \
    --data_usage 'test' \
"

if [ "$case" == "a" ] && [[ "$model_type"=="csrvcv4" ]]; then
    python_command+=" --seg_model_file 'model_seg_hcp_Unet_200epochs.pt' \
                      --model_file_wm_deformation '$model_file_wm_deformation' \
                      --model_file_wm_classification '$model_file_wm_classification' \
                      --model_file_gm_deformation '$model_file_gm_deformation' \
                      --model_file_gm_classification '$model_file_gm_classification' \
                      --model_dir '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv4_0/model/' \
                      --init_dir '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv4_0/init/'
    
    "
    echo 'av4'
elif [ "$case" == "b" ] && [[ "$model_type"=="csrvcv4" ]]; then
    python_command+=" --seg_model_file 'model_seg_hcp_Unet_200epochs.pt' \
                      --model_file_wm '$wm_model_file_combined' \
                      --model_file_gm '$gm_model_file_combined' \
                      --model_dir '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv4_0/model/' \
                      --init_dir '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv4_0/init/'
    
    "
    echo 'bv4'

elif [ "$case" == "c" ]; then
    python_command+=" --seg_model_file 'model_seg_hcp_Unet_200epochs.pt' \
                      --model_file_wm '$wm_model_file_combined' \
                      --model_file_gm '$gm_model_file_combined' \
                      --model_dir '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_csrf_gnn_0/model/' \
                      --init_dir '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_csrf_gnn_0/init/'
                      
    "
    echo 'c'

fi

# Run the Python script
echo "Running command:"
echo $python_command

eval $python_command
