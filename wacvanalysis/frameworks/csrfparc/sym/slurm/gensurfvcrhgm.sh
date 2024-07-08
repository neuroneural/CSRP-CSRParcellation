#!/bin/bash
#. /opt/miniconda3/bin/activate cortexode
. /opt/miniconda3/bin/activate csrf
cd /cortexode

version=1
gnn='gat'
gnn_layer=12
gat_head=1
model_file='model_vertex_classification_gm_hcp_rh_vc_v1_gnngat_layers12_heads1_280epochs.pt'
# Execute Python script with parameters set above
echo "Running configuration: Version $version, GNN $gnn, GNN Layers $gnn_layer, GAT Heads $gat_head"

python generateTestDiceAndSurfaces.py --model_file=$model_file --data_dir='/speedrun/cortexode-data-rp/' --model_dir='/cortexode/ckpts/hcp_vc_sym_gnn_0/model/' --result_dir='/cortexode/ckpts/hcp_vc_sym_gnn_0/result/' --model_type 'csrvc' --patience 'standard' --visualize 'yes' --version $version --gnn $gnn --gnn_layers $gnn_layer --gat_heads $gat_head --train_type='surfvc' --data_name='hcp'  --surf_hemi='rh' --surf_type='gm' --n_samples=150000 --tag='vc' --device='gpu' 
