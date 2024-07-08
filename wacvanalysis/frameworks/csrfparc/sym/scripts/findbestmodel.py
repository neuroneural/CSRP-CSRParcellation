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
    best_epoch = -1
    print(f"Parsing log file: {filepath}")
    with open(filepath, 'r') as file:
        for line in file:
            if 'validation dice' in line:
                try:
                    dice = float(re.search(r"validation dice:([\d\.]+)", line).group(1))
                    epoch = int(re.search(r"epoch:(\d+)", line).group(1))
                    if dice > best_validation_dice:
                        best_validation_dice = dice
                        best_epoch = epoch
                except Exception as e:
                    print(f"Error parsing line: {line} with error {e}")
    print(f"Best validation dice: {best_validation_dice} at epoch: {best_epoch}")
    return best_validation_dice, best_epoch

def find_model_file(logfile, model_dir, best_epoch):
    # Generate the model filename pattern
    params = extract_parameters_from_filename(logfile)
    if params:
        hemi, gnn_layers, heads = params
        model_pattern = f"model_vertex_classification_gm_hcp_{hemi}_vc_v1_gnngat_layers{gnn_layers}_heads{heads}_{best_epoch}epochs.pt"
        for filename in os.listdir(model_dir):
            if re.match(model_pattern, filename):
                return filename
    return None

def main(logs_directory):
    data = []

    for filename in os.listdir(logs_directory):
        if filename.endswith(".log"):
            print(f"Processing file: {filename}")
            params = extract_parameters_from_filename(filename)
            if params:
                hemi, gnn_layers, heads = params
                filepath = os.path.join(logs_directory, filename)
                best_validation_dice, best_epoch = parse_log_file(filepath)
                if best_validation_dice != -1.0 and best_epoch != -1:
                    model_file = find_model_file(filename, logs_directory, best_epoch)
                    if model_file:
                        data.append({
                            "hemi": hemi,
                            "surf_type": "gm",
                            "gnn_layers": gnn_layers,
                            "best_validation_dice": best_validation_dice,
                            "best_epoch": best_epoch,
                            "model_file": model_file,
                            "full_path": os.path.join(logs_directory, model_file)
                        })
                    else:
                        print(f"Model file for {filename} with best epoch {best_epoch} not found.")
                else:
                    print(f"No valid best_validation_dice or best_epoch found for {filename}.")
            else:
                print(f"Filename {filename} did not match the expected pattern.")

    if data:
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Sort the DataFrame by best_validation_dice in descending order
        df = df.sort_values(by='best_validation_dice', ascending=False)
        
        output_csv_path = os.path.join(
            '/data/users2/washbee/CortexODE-CSRFusionNet/wacvanalysis/frameworks/csrfparc/sym/csv',
            'best_validation_dice_scores.csv'
        )
        df.to_csv(output_csv_path, index=False)
        print(f"CSV file created: {output_csv_path}")
    else:
        print("No data to write to CSV.")

if __name__ == "__main__":
    logs_directory = "/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_vc_sym_gnn_0/model"
    main(logs_directory)
