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
    if len(sys.argv) < 4:
        print("Usage: python script.py <filename> <model_type> <solver> [<gnn_layers>]")
        sys.exit(1)

    filename = sys.argv[1]
    model_type = sys.argv[2]
    solver = sys.argv[3]
    gnn_layers = sys.argv[4] if len(sys.argv) > 4 and model_type == 'v2' else 'NA'
    
    surf_type, hemisphere = parse_filename(filename)
    data = extract_data_from_log(filename)
    output_csv_path = 'results.csv'
    write_to_csv(output_csv_path, hemisphere, surf_type, model_type, solver, gnn_layers, data)

if __name__ == '__main__':
    main()

