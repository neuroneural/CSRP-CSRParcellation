#!/bin/bash
. /opt/miniconda3/bin/activate csrf
cd /cortexode

RESULT_DIR="/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_dataloader_verify/result"

python ./wacvanalysis/verify_dataloader.py --data_dir='/speedrun/cortexode-data-rp/' --data_name='hcp'  --n_samples=150000 --tag='csrf' --device='gpu' --surf_hemi lh --surf_type gm --result_dir $RESULT_DIR
