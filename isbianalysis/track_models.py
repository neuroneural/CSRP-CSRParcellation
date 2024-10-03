import os
import re
import pandas as pd

# Define the path to your log files
log_dir = '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv3undirectedjoint_0/model'

# Define the output CSV file name
output_csv = 'validation_metrics.csv'

# Define possible modes
modes = ['_norecon_class', '_recon_noclass', '_recon_class']

# Define the two-level headers
# First level
header_level_1 = [
    '', '', '',  # surf_type, surf_hemi, GAT Layers
    'max epochs', 'max epochs', 'max epochs',
    'max validation dice', 'max validation dice epoch', 'max validation dice epoch filename',
    'max validation dice', 'max validation dice epoch', 'max validation dice epoch filename',
    'min reconstruction validation error', 'min reconstruction validation error (epoch)', 'min reconstruction validation error (filename)',
    'min reconstruction validation error', 'min reconstruction validation error (epoch)', 'min reconstruction validation error (filename)',
    'min reconstruction validation error', 'min reconstruction validation error (epoch)', 'min reconstruction validation error (filename)'
]

# Second level
header_level_2 = [
    'surf_type', 'surf_hemi', 'GAT Layers',
    '_norecon_class', '_recon_noclass', '_recon_class',
    '_norecon_class', '_norecon_class', '_norecon_class',
    '_recon_noclass', '_recon_noclass', '_recon_noclass',
    '_recon_class', '_recon_class', '_recon_class',
    '_recon_class', '_recon_class', '_recon_class'
]

# Since the header seems to repeat metrics for different modes, we'll adjust accordingly
# For clarity, we'll construct MultiIndex columns in pandas

# Define metrics per mode
metrics = {
    '_norecon_class': ['max_epochs', 'max_validation_dice', 'max_validation_dice_epoch', 'max_validation_dice_epoch_filename'],
    '_recon_noclass': ['max_epochs', 'max_validation_dice', 'max_validation_dice_epoch', 'max_validation_dice_epoch_filename'],
    '_recon_class': ['max_epochs', 'max_validation_dice', 'max_validation_dice_epoch', 'max_validation_dice_epoch_filename',
                     'min_recon_error', 'min_recon_error_epoch', 'min_recon_error_filename']
}

# Create MultiIndex for columns
tuples = []
# First three columns are surf_type, surf_hemi, GAT Layers with no sub-columns
tuples.extend([('', 'surf_type'), ('', 'surf_hemi'), ('', 'GAT Layers')])

# For each mode, add the metrics
for mode in modes:
    for metric in metrics[mode]:
        tuples.append((metric, mode))

# Create MultiIndex
index = pd.MultiIndex.from_tuples(tuples, names=['Metric', 'Mode'])

# Initialize an empty list to collect rows
rows_list = []

# Define possible values based on provided CSV data
surf_types = ['wm', 'gm']
surf_hemis = ['lh', 'rh']
gat_layers = [4, 8, 12]

# Populate the list with all combinations
for surf_type in surf_types:
    for surf_hemi in surf_hemis:
        for layer in gat_layers:
            row = {
                ('', 'surf_type'): surf_type,
                ('', 'surf_hemi'): surf_hemi,
                ('', 'GAT Layers'): layer
            }
            rows_list.append(row)

# Create DataFrame from the list of rows
df = pd.DataFrame(rows_list, columns=index)

# Function to parse log filenames
def parse_log_filename(filename):
    """
    Parses the log filename to extract parameters.
    Example filename:
    model_gm_hcp_lh_vc_v3_csrvc_layers12_sf0.1_euler_recon_noclass_551882_heads1.log
    """
    pattern = r'^model_(wm|gm)_hcp_(lh|rh)_.*?_layers(\d+).*?_(norecon_class|recon_noclass|recon_class)_.*?\.log$'
    match = re.match(pattern, filename)
    if match:
        surf_type = match.group(1)
        surf_hemi = match.group(2)
        gat_layers = int(match.group(3))
        mode = '_' + match.group(4)
        return surf_type, surf_hemi, gat_layers, mode
    else:
        return None

# Function to parse log contents
def parse_log_file(filepath):
    """
    Parses the log file to extract validation metrics.
    Returns a dictionary with extracted metrics.
    """
    metrics_extracted = {
        'max_epochs': None,
        'max_validation_dice': None,
        'max_validation_dice_epoch': None,
        'max_validation_dice_epoch_filename': None,
        'min_recon_error': None,
        'min_recon_error_epoch': None,
        'min_recon_error_filename': None
    }

    max_dice = -float('inf')
    max_dice_epoch = None

    min_recon_error = float('inf')
    min_recon_error_epoch = None

    max_epochs = 0

    with open(filepath, 'r') as file:
        for line in file:
            # Extract epoch number
            epoch_match = re.search(r'epoch:(\d+)', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                if epoch > max_epochs:
                    max_epochs = epoch

                # Check for dice validation error
                dice_match = re.search(r'dice validation error:([^\s]+)', line)
                if dice_match:
                    dice_value = dice_match.group(1)
                    if dice_value.lower() != 'nan':
                        try:
                            dice_value = float(dice_value)
                            if dice_value > max_dice:
                                max_dice = dice_value
                                max_dice_epoch = epoch
                        except ValueError:
                            pass  # Handle cases where conversion fails

                # Check for reconstruction validation error
                recon_match = re.search(r'reconstruction validation error:([^\s]+)', line)
                if recon_match:
                    recon_value = recon_match.group(1)
                    if recon_value.lower() != 'nan':
                        try:
                            recon_value = float(recon_value)
                            if recon_value < min_recon_error:
                                min_recon_error = recon_value
                                min_recon_error_epoch = epoch
                        except ValueError:
                            pass  # Handle cases where conversion fails

    # Assign extracted metrics
    metrics_extracted['max_epochs'] = max_epochs if max_epochs > 0 else None

    if max_dice != -float('inf'):
        metrics_extracted['max_validation_dice'] = max_dice
        metrics_extracted['max_validation_dice_epoch'] = max_dice_epoch

    if min_recon_error != float('inf'):
        metrics_extracted['min_recon_error'] = min_recon_error
        metrics_extracted['min_recon_error_epoch'] = min_recon_error_epoch

    return metrics_extracted

# Process each log file
for log_file in os.listdir(log_dir):
    if not log_file.endswith('.log'):
        continue  # Skip non-log files

    parsed = parse_log_filename(log_file)
    if not parsed:
        print(f"Skipping unrecognized log file format: {log_file}")
        continue

    surf_type, surf_hemi, gat_layers, mode = parsed

    # Full path to the log file
    log_path = os.path.join(log_dir, log_file)

    # Parse the log file
    metrics = parse_log_file(log_path)

    # Update the DataFrame
    # Locate the row matching surf_type, surf_hemi, gat_layers
    condition = (
        (df[('', 'surf_type')] == surf_type) &
        (df[('', 'surf_hemi')] == surf_hemi) &
        (df[('', 'GAT Layers')] == gat_layers)
    )

    if not condition.any():
        print(f"No matching row found for {surf_type}, {surf_hemi}, Layers {gat_layers}")
        continue

    row_index = df[condition].index[0]

    for metric, value in metrics.items():
        if value is not None:
            if metric.endswith('_filename'):
                # For filename metrics, assign the log filename
                df.at[row_index, (metric, mode)] = log_file
            else:
                df.at[row_index, (metric, mode)] = value

# Replace missing values with empty strings
df.fillna('', inplace=True)

# Save the DataFrame to CSV with two-level headers
df.to_csv(output_csv, index=False)

print(f"CSV file '{output_csv}' has been created successfully in the current directory.")
