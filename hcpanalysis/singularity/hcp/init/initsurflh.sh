#!/bin/bash
. /opt/miniconda3/bin/activate csrf
cd /cortexode

# python eval.py --test_type='init' --data_dir='/cortexode/bsnipanalysis/' --model_dir='/cortexode/ckpts/bsnip/bsnip_0/model/' --init_dir='/cortexode/ckpts/bsnip/bsnip_0/init/test/' --data_usage='test' --data_name='bsnip' --surf_hemi='lh' --tag='Unet' --device='gpu'

# python eval.py --test_type='init' --data_dir='/cortexode/bsnipanalysis/' --model_dir='/cortexode/ckpts/bsnip/bsnip_0/model/' --init_dir='/cortexode/ckpts/bsnip/bsnip_0/init/train/' --data_usage='train' --data_name='bsnip' --surf_hemi='lh' --tag='Unet' --device='gpu'

python eval.py --test_type='init' --data_dir='/cortexode/bsnipanalysis/' --model_dir='/cortexode/ckpts/bsnip/bsnip_0/model/' --init_dir='/cortexode/ckpts/bsnip/bsnip_0/init/valid/' --data_usage='valid' --data_name='bsnip' --surf_hemi='lh' --tag='Unet' --device='gpu'
