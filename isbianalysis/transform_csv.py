import pandas as pd
import csv
import re

# Read the CSV with two header rows
df = pd.read_csv('validation_metrics.csv', header=[0, 1])

# Optionally, print the columns to inspect their structure
print(df.columns)

# Initialize a list to hold rows for the new CSV
rows = []

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    # Access the columns using the correct MultiIndex keys
    surf_type = row[('Unnamed: 0_level_0', 'surf_type')]
    surf_hemi = row[('Unnamed: 1_level_0', 'surf_hemi')]
    gat_layers = row[('Unnamed: 2_level_0', 'GAT Layers')]

    # Iterate over each mode
    for mode in ['_norecon_class', '_recon_noclass', '_recon_class']:
        # Check if the mode has data
        best_model_filename = row.get(('best_model_filename', mode), '')
        if pd.notna(best_model_filename) and 'Log file does not exist' not in str(best_model_filename):
            # Extract additional parameters
            max_epochs = row.get(('max_epochs', mode), '')
            if mode in ['_norecon_class', '_recon_class']:
                epoch = row.get(('max_validation_dice_epoch', mode), '')
            else:
                epoch = row.get(('min_recon_error_epoch', mode), '')

            # Ensure epoch is an integer
            if pd.notna(epoch):
                try:
                    epoch = int(float(epoch))
                except ValueError:
                    print(f"Invalid epoch value '{epoch}' at index {index}. Setting epoch to empty string.")
                    epoch = ''
            else:
                epoch = ''

            # Extract random_number from the best_model_filename
            match = re.search(r'_(\d{6})\.pt$', str(best_model_filename))
            random_number = match.group(1) if match else ''

            # Determine reconstruction and classification flags
            reconstruction = 'False' if 'norecon' in mode else 'True'
            classification = 'False' if 'noclass' in mode else 'True'

            # Extract other parameters from the filename
            model_filename = str(best_model_filename)
            solver_match = re.search(r'_(euler|rk4)_', model_filename)
            solver = solver_match.group(1) if solver_match else 'euler'

            heads_match = re.search(r'_heads(\d+)_', model_filename)
            heads = heads_match.group(1) if heads_match else '1'

            version_match = re.search(r'_vc_(v\d+)_', model_filename)
            version = version_match.group(1) if version_match else 'v3'

            model_type_match = re.search(r'_(csrvc|other_model_types)_', model_filename)
            model_type = model_type_match.group(1) if model_type_match else 'csrvc'

            data_name_match = re.search(r'model_.*?_(\w+)_', model_filename)
            data_name = data_name_match.group(1) if data_name_match else 'hcp'

            # Create a row for the new CSV
            new_row = {
                'surf_type': surf_type,
                'data_name': data_name,
                'hemisphere': surf_hemi,
                'version': version,
                'model_type': model_type,
                'layers': gat_layers,
                'heads': heads,
                'epoch': epoch,
                'solver': solver,
                'reconstruction': reconstruction,
                'classification': classification,
                'random_number': random_number,
                'MODEL_FILE': best_model_filename,
                'MODEL_DIR': '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv3undirectedjoint_0/model/'
            }
            rows.append(new_row)

# Write the new CSV
new_csv_file = 'models_to_run.csv'
with open(new_csv_file, 'w', newline='') as csvfile:
    fieldnames = [
        'surf_type', 'data_name', 'hemisphere', 'version', 'model_type',
        'layers', 'heads', 'epoch', 'solver', 'reconstruction', 'classification',
        'random_number', 'MODEL_FILE', 'MODEL_DIR'
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"New CSV '{new_csv_file}' has been created with {len(rows)} models.")
