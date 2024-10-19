import csv
from collections import defaultdict

# Input and output CSV file paths
input_csv_file = 'models_to_run.csv'
output_csv_file = 'models_arguments.csv'

# Read the input CSV file
with open(input_csv_file, 'r', newline='') as infile:
    reader = csv.DictReader(infile)
    models = list(reader)

# Group models based on shared criteria
grouped_models = defaultdict(lambda: {'case': '', 'wm': {}, 'gm': {}})

# Process each model in the input CSV
for model in models:
    # Extract necessary fields from the model
    surf_type = model['surf_type']          # 'wm' or 'gm'
    data_name = model['data_name']
    hemisphere = model['hemisphere']
    layers = model['layers']
    solver = model['solver']
    gat_heads = model['heads']
    model_file = model['MODEL_FILE']
    model_dir = model['MODEL_DIR']
    reconstruction = model['reconstruction'] == 'True'
    classification = model['classification'] == 'True'
    version = model.get('version', '')              # Use .get to handle missing fields
    model_type = model.get('model_type', '')        # Use .get to handle missing fields
    # You can include additional fields if needed

    # Infer the case based on reconstruction and classification flags
    if reconstruction and classification:
        case = 'b'  # Combined model
    elif reconstruction != classification:
        case = 'a'  # Separate deformation or classification model
    else:
        # Skip models where both reconstruction and classification are False
        continue

    # Create a grouping key based on shared criteria, including the case
    key = (case, data_name, hemisphere, layers, solver, gat_heads)

    # Initialize the group if it doesn't exist
    group = grouped_models[key]
    group['case'] = case

    # Assign the model to the appropriate field in the group
    if case == 'a':
        # For case 'a', we need separate deformation and classification models
        if surf_type == 'wm':
            if reconstruction and not classification:
                group['wm']['deformation'] = {'model_file': model_file, 'model_dir': model_dir}
            elif classification and not reconstruction:
                group['wm']['classification'] = {'model_file': model_file, 'model_dir': model_dir}
        elif surf_type == 'gm':
            if reconstruction and not classification:
                group['gm']['deformation'] = {'model_file': model_file, 'model_dir': model_dir}
            elif classification and not reconstruction:
                group['gm']['classification'] = {'model_file': model_file, 'model_dir': model_dir}
    elif case == 'b':
        # For case 'b', we need combined models
        if surf_type == 'wm':
            group['wm']['combined'] = {'model_file': model_file, 'model_dir': model_dir}
        elif surf_type == 'gm':
            group['gm']['combined'] = {'model_file': model_file, 'model_dir': model_dir}

# Define the field names for the output CSV
output_fieldnames = [
    # Common fields
    'case',
    'data_name',
    'surf_hemi',
    'gnn_layers',
    'solver',
    'gat_heads',
    # WM model fields
    'wm_model_dir',
    'model_file_wm',
    'model_file_wm_deformation',
    'model_file_wm_classification',
    # GM model fields
    'gm_model_dir',
    'model_file_gm',
    'model_file_gm_deformation',
    'model_file_gm_classification'
]

# Open the output CSV file for writing
with open(output_csv_file, 'w', newline='') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
    writer.writeheader()

    # Iterate over the grouped models and write to the output CSV
    for key, group in grouped_models.items():
        case, data_name, hemisphere, layers, solver, gat_heads = key

        # Initialize the output row with common fields
        output_row = {
            'case': case,
            'data_name': data_name,
            'surf_hemi': hemisphere,
            'gnn_layers': layers,
            'solver': solver,
            'gat_heads': gat_heads,
            # Initialize model fields to empty strings
            'wm_model_dir': '',
            'model_file_wm': '',
            'model_file_wm_deformation': '',
            'model_file_wm_classification': '',
            'gm_model_dir': '',
            'model_file_gm': '',
            'model_file_gm_deformation': '',
            'model_file_gm_classification': ''
        }

        # Process WM models
        wm_models = group['wm']
        if case == 'a':
            # For case 'a', we expect separate deformation and classification models
            if 'deformation' in wm_models:
                output_row['wm_model_dir'] = wm_models['deformation']['model_dir']
                output_row['model_file_wm_deformation'] = wm_models['deformation']['model_file']
            if 'classification' in wm_models:
                output_row['wm_model_dir'] = wm_models['classification']['model_dir']
                output_row['model_file_wm_classification'] = wm_models['classification']['model_file']
        elif case == 'b':
            # For case 'b', we expect a combined model
            if 'combined' in wm_models:
                output_row['wm_model_dir'] = wm_models['combined']['model_dir']
                output_row['model_file_wm'] = wm_models['combined']['model_file']

        # Process GM models
        gm_models = group['gm']
        if case == 'a':
            # For case 'a', we expect separate deformation and classification models
            if 'deformation' in gm_models:
                output_row['gm_model_dir'] = gm_models['deformation']['model_dir']
                output_row['model_file_gm_deformation'] = gm_models['deformation']['model_file']
            if 'classification' in gm_models:
                output_row['gm_model_dir'] = gm_models['classification']['model_dir']
                output_row['model_file_gm_classification'] = gm_models['classification']['model_file']
        elif case == 'b':
            # For case 'b', we expect a combined model
            if 'combined' in gm_models:
                output_row['gm_model_dir'] = gm_models['combined']['model_dir']
                output_row['model_file_gm'] = gm_models['combined']['model_file']

        # Optional: Log missing models
        missing_models = []
        if case == 'a':
            if not (output_row['model_file_wm_deformation'] or output_row['model_file_wm_classification']):
                missing_models.append('WM deformation/classification')
            if not (output_row['model_file_gm_deformation'] or output_row['model_file_gm_classification']):
                missing_models.append('GM deformation/classification')
        elif case == 'b':
            if not output_row['model_file_wm']:
                missing_models.append('WM combined')
            if not output_row['model_file_gm']:
                missing_models.append('GM combined')

        if missing_models:
            print(f"Warning: Missing models for case {key}: {', '.join(missing_models)}")

        # Write the row to the CSV (including incomplete cases with empty fields)
        writer.writerow(output_row)
