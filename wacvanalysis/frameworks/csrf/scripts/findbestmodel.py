import os
import re
import pandas as pd
import shutil

def extract_parameters_from_filename(filename):
    # Regular expression for extracting necessary parts from filenames
    pattern = re.compile(r"model_(gm|wm)_(hcp)_(lh|rh)_csrf_v(\d+)_gnn(gat)_layers(\d+)_sf(\d+\.\d+)_(euler)_heads(\d+)\.log") #added gat
    match = pattern.match(filename)
    if match:
        # Extract matched groups
        matched_params = match.groups()
        return matched_params
    return None

def parse_log_file(filepath):
    best_validation_error = float('inf')
    best_epoch = -1
    print(f"Parsing log file: {filepath}")
    with open(filepath, 'r') as file:
        for line in file:
            if 'validation error' in line:
                print(line)
                try:
                    epoch = int(re.search(r"epoch:(\d+)", line).group(1))
                    validation_error = float(re.search(r"validation error:([\d\.]+)", line).group(1))
                    if epoch <= 200 and validation_error < best_validation_error:
                        best_validation_error = validation_error
                        best_epoch = epoch
                except Exception as e:
                    print(f"Error parsing line: {line} with error {e}")
    return best_epoch, best_validation_error

def main(logs_directory):
    data = []
    model_dir = "../model"
    os.makedirs(model_dir, exist_ok=True)
    
    best_models = {}

    for filename in os.listdir(logs_directory):
        if filename.endswith(".log"):
            print(f"Processing file: {filename}")
            params = extract_parameters_from_filename(filename)
            if params:
                print('params')
                print(params)
                surf_type, dataset, hemi, version, gnn, gnn_layers, scaling_factor, solver, heads = params
                filepath = os.path.join(logs_directory, filename)
                best_epoch, best_validation_error = parse_log_file(filepath)
                print('best_epoch',best_epoch)
                print('best_validation_error',best_validation_error)
                if best_epoch != -1:
                    key = (surf_type, hemi)
                    print('key')
                    print(key)
                    if key not in best_models or best_models[key]['best_validation_error'] > best_validation_error:
                        best_models[key] = {
                            "surf_type": surf_type,
                            "dataset": dataset,
                            "hemi": hemi,
                            "version": version,
                            "gnn_layers": gnn_layers,
                            "scaling_factor": scaling_factor,
                            "solver": solver,
                            "heads": heads,
                            "best_epoch": best_epoch,
                            "best_validation_error": best_validation_error,
                            "logfile": filename
                        }

    for key, model_info in best_models.items():
        #real model_wm_hcp_rh_csrf_v3_gnngat_layers4_sf0.1_heads1_50epochs_euler.pt
        model_filename = f"model_{model_info['surf_type']}_{model_info['dataset']}_{model_info['hemi']}_csrf_v{model_info['version']}_gnngat_layers{model_info['gnn_layers']}_sf{model_info['scaling_factor']}_heads{model_info['heads']}_{model_info['best_epoch']}epochs_{model_info['solver']}.pt"
        model_path = os.path.abspath(os.path.join(logs_directory, model_filename))
        print('key,model_info',key,model_info)
        print(model_path)
        if os.path.exists(model_path):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            shutil.copy(model_path, model_dir)
            data.append({
                "surf_type": model_info['surf_type'],
                "dataset": model_info['dataset'],
                "hemi": model_info['hemi'],
                "version": model_info['version'],
                "gnn_layers": model_info['gnn_layers'],
                "scaling_factor": model_info['scaling_factor'],
                "solver": model_info['solver'],
                "heads": model_info['heads'],
                "best_epoch": model_info['best_epoch'],
                "best_validation_error": model_info['best_validation_error'],
                "logfile": model_info['logfile'],
                "model_path": model_path
            })

    df = pd.DataFrame(data)
    output_csv_path = '/data/users2/washbee/CortexODE-CSRFusionNet/wacvanalysis/frameworks/csrf/csvs/best_validation_errors.csv'
    df.to_csv(output_csv_path, index=False)
    print(f"CSV file created: {output_csv_path}")

if __name__ == "__main__":
    logs_directory = "/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_csrf_gnn_2/model"  # Replace with your logs directory
    main(logs_directory)
