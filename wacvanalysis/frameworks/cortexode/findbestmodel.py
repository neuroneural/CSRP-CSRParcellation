import os
import re
import pandas as pd
import shutil

def extract_parameters_from_filename(filename):
    # Regular expression for extracting necessary parts from filenames
    pattern = re.compile(r"model_(gm|wm)_(hcp)_(lh|rh)_csrf_v(\d+)_gnnbaseline_sf(\d+\.\d+)_(euler)\.log")
    match = pattern.match(filename)
    if match:
        # Extract matched groups
        matched_params = match.groups()
        return matched_params
    return None

def parse_log_file(filepath):
    best_validation_error = float('inf')
    best_epoch = -1
    with open(filepath, 'r') as file:
        for line in file:
            if 'validation error' in line:
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
    model_dir = "./model"
    os.makedirs(model_dir, exist_ok=True)
    
    for filename in os.listdir(logs_directory):
        if filename.endswith(".log"):
            params = extract_parameters_from_filename(filename)
            if params:
                surf_type, dataset, hemi, version, scaling_factor, solver = params
                filepath = os.path.join(logs_directory, filename)
                best_epoch, best_validation_error = parse_log_file(filepath)
                if best_epoch != -1:
                    model_filename = f"model_{surf_type}_{dataset}_{hemi}_csrf_v{version}_gnnbaseline_sf{scaling_factor}_{best_epoch}epochs_{solver}.pt"
                    model_path = os.path.abspath(os.path.join(logs_directory, model_filename))
                    if os.path.exists(model_path):
                        shutil.copy(model_path, model_dir)
                        data.append({
                            "surf_type": surf_type,
                            "dataset": dataset,
                            "hemi": hemi,
                            "version": version,
                            "scaling_factor": scaling_factor,
                            "solver": solver,
                            "best_epoch": best_epoch,
                            "best_validation_error": best_validation_error,
                            "model_path": model_path
                        })

    df = pd.DataFrame(data)
    output_csv_path = '/data/users2/washbee/CortexODE-CSRFusionNet/wacvanalysis/frameworks/cortexode/csvs/best_validation_errors.csv'
    df.to_csv(output_csv_path, index=False)
    print(f"CSV file created: {output_csv_path}")

if __name__ == "__main__":
    logs_directory = "/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/hcp_csrf_gnn_0/model"  # Replace with your logs directory
    main(logs_directory)
