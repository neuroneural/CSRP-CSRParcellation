import torch
from csrvcv2 import CSRVCV2
# Instantiate the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
csrvcv2_wm = CSRVCV2(dim_h=128,
                     kernel_size=5,
                     n_scale=3,
                     sf=0.1,
                     gnn_layers=6,
                     use_gcn=False,
                     gat_heads=1,
                     num_classes=36).to(device)


# Load the saved model into a new instance
csrvcv2_wm_loaded = CSRVCV2(dim_h=128,
                            kernel_size=5,
                            n_scale=3,
                            sf=0.1,
                            gnn_layers=6,
                            use_gcn=False,
                            gat_heads=1,
                            num_classes=36).to(device)

csrvcv2_wm_loaded.load_state_dict(torch.load('/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi_gnn_1/model/model_gm_hcp_rh_vc_v2_csrvc_layers6_sf0.1_heads1_10epochs_euler.pt', map_location=device))


# Compare the parameters of the original model and the reloaded model
def compare_models(model1, model2):
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        print(name1,name2,param1.shape,param2.shape)
        if param1.shape != param2.shape:
            print(param1.shape,param2.shape)
            print(f"Mismatch found in layer: {name1}")
            return False
    print("The models are identical.")
    return True

# Compare the original model and the loaded model
compare_models(csrvcv2_wm, csrvcv2_wm_loaded)
