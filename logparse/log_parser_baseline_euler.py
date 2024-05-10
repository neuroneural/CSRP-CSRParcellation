import sys
import re
import csv
import os

def parse_filename(filename):
    # Extracting surf_type and hemisphere from the filename
    parts = filename.split('_')
    surf_type = next((part for part in parts if part in ['gm', 'wm']), None)
    hemisphere = next((part for part in parts if part in ['lh', 'rh']), None)
    return surf_type, hemisphere

def extract_data_from_log(filepath):
    # Extract epoch and validation error from the log file
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            if 'validation error' in line:
                epoch = re.search(r'epoch:(\d+)', line)
                error = re.search(r'validation error:(\d+\.\d+)', line)
                if epoch and error:
                    data.append((int(epoch.group(1)), float(error.group(1))))
    return data

def write_to_csv(filepath, hemisphere, surf_type, model_type, solver, gnn_layers, data):
    # Check if file exists and if headers are needed
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['hemisphere', 'surf_type', 'model_type', 'solver', 'gnn_layers', 'epochs', 'validation_error'])
        for epochs, validation_error in data:
            writer.writerow([hemisphere, surf_type, model_type, solver, gnn_layers, epochs, validation_error])

def main():
    # Expecting filename, model_type, solver, and optionally gnn_layers
    
    basedir='/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/experiment_5/model/'
    filenames = ['model_gm_adni_lh_exp5.log', 'model_gm_adni_rh_exp5.log',  'model_wm_adni_lh_exp5.log' , 'model_wm_adni_rh_exp5.log']
    model_type = 'baseline'
    solver = 'euler'
    gnn_layers = 'NA'
    output_csv_path = 'results_experiment5_baseline_euler_NA.csv'
        
    for filename in filenames:
        filename = os.path.join(basedir, filename)
        surf_type, hemisphere = parse_filename(filename)
        data = extract_data_from_log(filename)
        write_to_csv(output_csv_path, hemisphere, surf_type, model_type, solver, gnn_layers, data)

if __name__ == '__main__':
    main()

