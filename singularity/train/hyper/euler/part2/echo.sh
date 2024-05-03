#!/bin/bash

# Set these variables based on the specific run
gnn_layer=3  # Can be 2, 3, or 4
st='wm'  # Can be 'wm' or 'gm'
sh='rh'  # Can be 'lh' or 'rh'

# Selecting the model file based on the number of gnn_layer, st type, and sh
if [ "$gnn_layer" -eq 4 ]; then
    if [ "$st" == "gm" ] && [ "$sh" == "lh" ]; then
        model_file="model_gm_adni_lh_csrf_v2_gnngat_gnn_layer4_sf0.1_heads1_130epochs_euler.pt"
    elif [ "$st" == "wm" ] && [ "$sh" == "lh" ]; then
        model_file="model_wm_adni_lh_csrf_v2_gnngat_gnn_layer4_sf0.1_heads1_110epochs_euler.pt"
    elif [ "$st" == "gm" ] && [ "$sh" == "rh" ]; then
        model_file="model_gm_adni_rh_csrf_v2_gnngat_gnn_layer4_sf0.1_heads1_130epochs_euler.pt"
    elif [ "$st" == "wm" ] && [ "$sh" == "rh" ]; then
        model_file="model_wm_adni_rh_csrf_v2_gnngat_gnn_layer4_sf0.1_heads1_110epochs_euler.pt"
    fi
elif [ "$gnn_layer" -eq 3 ]; then
    if [ "$st" == "gm" ] && [ "$sh" == "lh" ]; then
        model_file="model_gm_adni_lh_csrf_v2_gnngat_gnn_layer3_sf0.1_heads1_160epochs_euler.pt"
    elif [ "$st" == "wm" ] && [ "$sh" == "lh" ]; then
        model_file="model_wm_adni_lh_csrf_v2_gnngat_gnn_layer3_sf0.1_heads1_130epochs_euler.pt"
    elif [ "$st" == "gm" ] && [ "$sh" == "rh" ]; then
        model_file="model_gm_adni_rh_csrf_v2_gnngat_gnn_layer3_sf0.1_heads1_160epochs_euler.pt"
    elif [ "$st" == "wm" ] && [ "$sh" == "rh" ]; then
        model_file="model_wm_adni_rh_csrf_v2_gnngat_gnn_layer3_sf0.1_heads1_130epochs_euler.pt"
    fi
elif [ "$gnn_layer" -eq 2 ]; then
    if [ "$st" == "gm" ] && [ "$sh" == "lh" ]; then
        model_file="model_gm_adni_lh_csrf_v2_gnngat_gnn_layer2_sf0.1_heads1_190epochs_euler.pt"
    elif [ "$st" == "wm" ] && [ "$sh" == "lh" ]; then
        model_file="model_wm_adni_lh_csrf_v2_gnngat_gnn_layer2_sf0.1_heads1_150epochs_euler.pt"
    elif [ "$st" == "gm" ] && [ "$sh" == "rh" ]; then
        model_file="model_gm_adni_rh_csrf_v2_gnngat_gnn_layer2_sf0.1_heads1_200epochs_euler.pt"
    elif [ "$st" == "wm" ] && [ "$sh" == "rh" ]; then
        model_file="model_wm_adni_rh_csrf_v2_gnngat_gnn_layer2_sf0.1_heads1_150epochs_euler.pt"
    fi
fi

# Extract epoch number from the model file name and increment it
epoch_num=$(echo $model_file | grep -o '[0-9]*epochs' | grep -o '[0-9]*')
start_epoch=$((epoch_num + 1))

echo "Selected model file: $model_file"
echo "Starting epoch: $start_epoch"
