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
wm_model_file_deformation="$9"
wm_model_file_classification="${10}"
gm_model_dir="${11}"
gm_model_file_combined="${12}"
gm_model_file_deformation="${13}"
gm_model_file_classification="${14}"

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
echo "wm_model_file_deformation: $wm_model_file_deformation"
echo "wm_model_file_classification: $wm_model_file_classification"
echo "gm_model_dir: $gm_model_dir"
echo "gm_model_file_combined: $gm_model_file_combined"
echo "gm_model_file_deformation: $gm_model_file_deformation"
echo "gm_model_file_classification: $gm_model_file_classification"
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
    --seg_model_file 'model_seg_hcp_Unet_200epochs.pt' \
    --result_dir '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv4_0/result/' \
    --wm_model_dir '$wm_model_dir' \
    --gm_model_dir '$gm_model_dir' \
    --model_type '$model_type'
"

# Add case-specific arguments
if [ "$case" == "a" ] && [[ "$model_type"=="csrvcv3" ]]; then
    python_command+=" --model_file_wm_deformation '$wm_model_file_deformation' \
                      --model_file_wm_classification '$wm_model_file_classification' \
                      --model_file_gm_deformation '$gm_model_file_deformation' \
                      --model_file_gm_classification '$gm_model_file_classification' \
                      --model_dir '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv3undirectedjoint_0/model/' \
                      --init_dir '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv3undirectedjoint_0/init/' \
    "
    echo 'av3'
elif [ "$case" == "b" ] && [[ "$model_type"=="csrvcv3" ]]; then
    python_command+=" --model_file_wm '$wm_model_file_combined' \
                      --model_file_gm '$gm_model_file_combined' \
                      --model_dir '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv3undirectedjoint_0/model/' \
                      --init_dir '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv3undirectedjoint_0/init/' \
    "
    echo 'bv3'
elif [ "$case" == "a" ] && [[ "$model_type"=="csrvcv4" ]]; then
    python_command+=" --model_file_wm_deformation '$wm_model_file_deformation' \
                      --model_file_wm_classification '$wm_model_file_classification' \
                      --model_file_gm_deformation '$gm_model_file_deformation' \
                      --model_file_gm_classification '$gm_model_file_classification' \
                      --model_dir '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi/isbi_gnnv4_0/model/' \
                      --init_dir '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi/isbi_gnnv4_0/init/' \
    "
    echo 'av4'
elif [ "$case" == "b" ] && [[ "$model_type"=="csrvcv4" ]]; then
    python_command+=" --model_file_wm '$wm_model_file_combined' \
                      --model_file_gm '$gm_model_file_combined' \
                      --model_dir '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi/isbi_gnnv4_0/model/' \
                      --init_dir '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv4_0/init/' \
    "
    echo 'bv4'
elif [ "$case" == "c" ]; then
    python_command+=" --model_file_wm '$wm_model_file_deformation' \
                      --model_file_gm '$gm_model_file_deformation' \
                      --model_type 'cortexode' \
                      --model_dir '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv3undirectedjoint_0/model/' \
                      --init_dir '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv3undirectedjoint_0/init/' \
    "
    echo 'c'
else
    echo "Unknown case: $case"
    exit 1
fi

# Run the Python script
echo "Running command:"
echo $python_command

eval $python_command
