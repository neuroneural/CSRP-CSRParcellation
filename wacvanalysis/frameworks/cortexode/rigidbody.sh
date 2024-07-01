#!/bin/bash

# Activate the conda environment
. /opt/miniconda3/bin/activate csrf

# Change to the cortexode directory
cd /cortexode

python /cortexode/wacvanalysis/frameworks/rigidbody_affine_ts.py \
    --data_dir "/data/users2/washbee/speedrun/cortexode-data-rp/" \
    --init_dir "/cortexode/ckpts/hcp_csrf_gnn_0/init/" \
    --result_dir "/cortexode/wacvanalysis/frameworks/cortexode/results/" \
    --model_type "cortexode" \
    --data_name "hcp" \
    --surf_type "wm" \
    --surf_hemi "lh" 

python /cortexode/wacvanalysis/frameworks/rigidbody_affine_ts.py \
    --data_dir "/data/users2/washbee/speedrun/cortexode-data-rp/" \
    --init_dir "/cortexode/ckpts/hcp_csrf_gnn_0/init/" \
    --result_dir "/cortexode/wacvanalysis/frameworks/cortexode/results/" \
    --model_type "cortexode" \
    --data_name "hcp" \
    --surf_type "wm" \
    --surf_hemi "rh" 
    
python /cortexode/wacvanalysis/frameworks/rigidbody_affine_ts.py \
--data_dir "/data/users2/washbee/speedrun/cortexode-data-rp/" \
--init_dir "/cortexode/ckpts/hcp_csrf_gnn_0/init/" \
--result_dir "/cortexode/wacvanalysis/frameworks/cortexode/results/" \
--model_type "cortexode" \
--data_name "hcp" \
--surf_type "gm" \
--surf_hemi "lh" 

python /cortexode/wacvanalysis/frameworks/rigidbody_affine_ts.py \
    --data_dir "/data/users2/washbee/speedrun/cortexode-data-rp/" \
    --init_dir "/cortexode/ckpts/hcp_csrf_gnn_0/init/" \
    --result_dir "/cortexode/wacvanalysis/frameworks/cortexode/results/" \
    --model_type "cortexode" \
    --data_name "hcp" \
    --surf_type "gm" \
    --surf_hemi "rh" 