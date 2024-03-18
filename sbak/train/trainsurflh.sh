#!/bin/bash
. /opt/miniconda3/bin/activate cortexode
cd /cortexode

python train.py --train_type='surf' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='/cortexode/ckpts/experiment_4/model/' --init_dir='/cortexode/ckpts/experiment_4/init/' --data_name='hcp'  --surf_hemi='lh' --surf_type='wm' --n_epochs=400 --n_samples=150000 --tag='exp4' --solver='euler' --step_size=0.1 --device='gpu' &
PID=$!
python train.py --train_type='surf' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='/cortexode/ckpts/experiment_4/model/' --init_dir='/cortexode/ckpts/experiment_4/init/' --data_name='hcp'  --surf_hemi='lh' --surf_type='gm' --n_epochs=400 --n_samples=150000 --tag='exp4' --solver='euler' --step_size=0.1 --device='gpu' &

wait $!
wait PID


