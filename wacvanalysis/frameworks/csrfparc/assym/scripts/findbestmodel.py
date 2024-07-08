import os
import re
import pandas as pd

def extract_parameters_from_filename(filename):
    # Regular expression for extracting necessary parts from filenames
    pattern = re.compile(r"model_vertex_classification_gm_hcp_(lh|rh)_vc_v1_gnngat_layers(\d+)_heads(\d+)\.log")
    match = pattern.match(filename)
    if match:
        # Extract matched groups
        hemi, gnn_layers, heads = match.groups()
        return hemi, int(gnn_layers), int(heads)
    return None

def parse_log_file(filepath):
    best_validation_dice = -1.0
    print(f"Parsing log file: {filepath}")
    with open(filepath, 'r') as file:
        for line in file:
            if 'validation dice' in line:
                try:
                    dice = float(re.search(r"validation dice:([\d\.]+)", line).group(1))
                    if dice > best_validation_dice:
                        best_validation_dice = dice
                except Exception as e:
                    print(f"Error parsing line: {line} with error {e}")
    return best_validation_dice

def main(logs_directory):
    data = []

    for filename in os.listdir(logs_directory):
        if filename.endswith(".log"):
            print(f"Processing file: {filename}")
            params = extract_parameters_from_filename(filename)
            if params:
                hemi, gnn_layers, heads = params
                filepath = os.path.join(logs_directory, filename)
                best_validation_dice = parse_log_file(filepath)
                if best_validation_dice != -1.0:
                    data.append({
                        "hemi": hemi,
                        "surf_type": "gm",
                        "gnn_layers": gnn_layers,
                        "best_validation_dice": best_validation_dice,
                        "logfile": filename
                    })

    df = pd.DataFrame(data)
    
    output_csv_path = os.path.join(
        '/data/users2/washbee/CortexODE-CSRFusionNet/wacvanalysis/frameworks/csrfparc/assym/csv',
        'best_validation_dice_scores.csv'
        )
    df.to_csv(output_csv_path, index=False)
    print(f"CSV file created: {output_csv_path}")

if __name__ == "__main__":
    logs_directory = "/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_vc_assym_gnn_2/model"
    main(logs_directory)
