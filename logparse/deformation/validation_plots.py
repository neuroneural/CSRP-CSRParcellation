import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to extract data from log files
def extract_data_from_log(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'epoch:(\d+), validation error:(\d+\.\d+)', line)
            if match:
                epoch = int(match.group(1))
                val_error = float(match.group(2))
                data.append((epoch, val_error))
    return data

# Directories and files information
log_info = [
    {
        "description": "euler baseline, no gnn",
        "directory": "/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_csrf_gnn_0/model/",
        "files": [
            "model_gm_hcp_lh_csrf_v0_gnnbaseline_sf0.1_euler.log",
            "model_gm_hcp_rh_csrf_v0_gnnbaseline_sf0.1_euler.log",
            "model_wm_hcp_lh_csrf_v0_gnnbaseline_sf0.1_euler.log",
            "model_wm_hcp_rh_csrf_v0_gnnbaseline_sf0.1_euler.log"
        ],
        "solver": "euler",
        "gnn": False,
        "ablation": None
    },
    {
        "description": "rk4 baseline, no gnn",
        "directory": "/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_csrf_gnn_0/model/",
        "files": [
            "model_gm_hcp_lh_csrf_v0_gnnbaseline_sf0.1_rk4.log",
            "model_gm_hcp_rh_csrf_v0_gnnbaseline_sf0.1_rk4.log",
            "model_wm_hcp_lh_csrf_v0_gnnbaseline_sf0.1_rk4.log",
            "model_wm_hcp_rh_csrf_v0_gnnbaseline_sf0.1_rk4.log"
        ],
        "solver": "rk4",
        "gnn": False,
        "ablation": None
    },
    {
        "description": "(ablate mlp layers) 1 linear layer, gnn layers 2",
        "directory": "/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_csrf_gnn_1/model/",
        "files": [
            "model_gm_hcp_lh_csrf_v2_gnngat_layers2_sf0.1_euler_heads1.log",
            "model_gm_hcp_rh_csrf_v2_gnngat_layers2_sf0.1_euler_heads1.log",
            "model_wm_hcp_lh_csrf_v2_gnngat_layers2_sf0.1_euler_heads1.log",
            "model_wm_hcp_rh_csrf_v2_gnngat_layers2_sf0.1_euler_heads1.log"
        ],
        "solver": "euler",
        "gnn": True,
        "ablation": "mlp"
    },
    {
        "description": "(Full model extension) Wide MLP, gnn layers 2",
        "directory": "/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_csrf_gnn_2/model/",
        "files": [
            "model_gm_hcp_lh_csrf_v3_gnngat_layers2_sf0.1_euler_heads1.log",
            "model_gm_hcp_rh_csrf_v3_gnngat_layers2_sf0.1_euler_heads1.log",
            "model_wm_hcp_lh_csrf_v3_gnngat_layers2_sf0.1_euler_heads1.log",
            "model_wm_hcp_rh_csrf_v3_gnngat_layers2_sf0.1_euler_heads1.log"
        ],
        "solver": "euler",
        "gnn": True,
        "ablation": "full_model_2_layers"
    },
    {
        "description": "(Full model extension) Wide MLP, gnn layers 3",
        "directory": "/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_csrf_gnn_2/model/",
        "files": [
            "model_gm_hcp_lh_csrf_v3_gnngat_layers3_sf0.1_euler_heads1.log",
            "model_gm_hcp_rh_csrf_v3_gnngat_layers3_sf0.1_euler_heads1.log",
            "model_wm_hcp_lh_csrf_v3_gnngat_layers3_sf0.1_euler_heads1.log",
            "model_wm_hcp_rh_csrf_v3_gnngat_layers3_sf0.1_euler_heads1.log"
        ],
        "solver": "euler",
        "gnn": True,
        "ablation": "full_model_3_layers"
    },
    {
        "description": "(Full model extension) Wide MLP, gnn layers 4",
        "directory": "/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_csrf_gnn_2/model/",
        "files": [
            "model_gm_hcp_lh_csrf_v3_gnngat_layers4_sf0.1_euler_heads1.log",
            "model_gm_hcp_rh_csrf_v3_gnngat_layers4_sf0.1_euler_heads1.log",
            "model_wm_hcp_lh_csrf_v3_gnngat_layers4_sf0.1_euler_heads1.log",
            "model_wm_hcp_rh_csrf_v3_gnngat_layers4_sf0.1_euler_heads1.log"
        ],
        "solver": "euler",
        "gnn": True,
        "ablation": "full_model_4_layers"
    },
    {
        "description": "(ablate position of vertices) Wide MLP, gnn layers 2",
        "directory": "/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_csrf_gnn_3/model/",
        "files": [
            "model_gm_hcp_lh_csrf_v4_gnngat_layers2_sf0.1_euler_heads1.log",
            "model_gm_hcp_rh_csrf_v4_gnngat_layers2_sf0.1_euler_heads1.log",
            "model_wm_hcp_lh_csrf_v4_gnngat_layers2_sf0.1_euler_heads1.log",
            "model_wm_hcp_rh_csrf_v4_gnngat_layers2_sf0.1_euler_heads1.log"
        ],
        "solver": "euler",
        "gnn": True,
        "ablation": "position"
    },
    {
        "description": "(ablate normals of vertices) Wide MLP, gnn layers 2, normals ablated",
        "directory": "/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_csrf_gnn_4/model/",
        "files": [
            "model_gm_hcp_lh_csrf_v5_gnngat_layers2_sf0.1_euler_heads1.log",
            "model_gm_hcp_rh_csrf_v5_gnngat_layers2_sf0.1_euler_heads1.log",
            "model_wm_hcp_lh_csrf_v5_gnngat_layers2_sf0.1_euler_heads1.log",
            "model_wm_hcp_rh_csrf_v5_gnngat_layers2_sf0.1_euler_heads1.log"
        ],
        "solver": "euler",
        "gnn": True,
        "ablation": "normals"
    }
]

# Initialize a DataFrame to hold all data
columns = ["epoch", "validation_error", "model", "hemisphere", "surface", "solver"]
data_list = []

# Read data from all logs and append to DataFrame
for log_set in log_info:
    for log_file in log_set["files"]:
        data = extract_data_from_log(os.path.join(log_set["directory"], log_file))
        model = log_set["description"]
        solver = log_set["solver"]
        hemisphere = "lh" if "lh" in log_file else "rh"
        surface = "gm" if "gm" in log_file else "wm"
        
        for epoch, val_error in data:
            data_list.append({
                "epoch": epoch,
                "validation_error": val_error,
                "model": model,
                "hemisphere": hemisphere,
                "surface": surface,
                "solver": solver
            })

df = pd.DataFrame(data_list, columns=columns)

# Plotting
g = sns.FacetGrid(df, row="surface", col="hemisphere", hue="model", margin_titles=True)
g.map(sns.lineplot, "epoch", "validation_error")

# Set y-axis to log scale and cap at 100
g.set(yscale="log", ylim=(.025, .04))

# Adjustments for better visualization
g.add_legend()
g.set_axis_labels("Epoch", "Validation Error")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Validation Error vs Epoch for Various Models and Configurations')

# Save plots
plt.savefig("validation_error_vs_epoch.png")
plt.savefig("validation_error_vs_epoch.svg")
plt.show()
