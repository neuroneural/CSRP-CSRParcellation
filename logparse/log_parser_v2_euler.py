import csv

def transform_and_append_data(source_csv, target_csv):
    # Open the source CSV and read data
    with open(source_csv, mode='r') as infile:
        reader = csv.DictReader(infile)
        transformed_rows = []

        for row in reader:
            # Map fields from new CSV to the expected format
            transformed_row = {
                'hemisphere': row['surf_hemi'],
                'surf_type': row['surf_type'],
                'model_type': 'v2',  # Since all data seems to belong to version 2
                'solver': row['solver'],
                'gnn_layers': row['gnn_layers'],
                'epochs': row['epoch'],
                'validation_error': row['validation_error']
            }
            transformed_rows.append(transformed_row)

    # Open the target CSV to append data
    with open(target_csv, mode='a', newline='') as outfile:
        fieldnames = ['hemisphere', 'surf_type', 'model_type', 'solver', 'gnn_layers', 'epochs', 'validation_error']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        # Check if file is empty to write headers
        outfile.seek(0, 2)  # Move the cursor to the end of the file
        if outfile.tell() == 0:
            writer.writeheader()

        # Append transformed data
        writer.writerows(transformed_rows)

# Usage
source_csv_path = '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/exp_vc_gnn_0/model/'
target_csv_path = 'results_exp_csrf_gnn_3.csv'
transform_and_append_data(source_csv_path, target_csv_path)
