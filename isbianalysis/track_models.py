# python track_models.py \
#     --max_epoch_norecon_class 50 \
#     --max_epoch_recon_noclass 60 \
#     --max_epoch_recon_class 70 \
#     --log_dir /data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv3undirectedjoint_0/model \
#     --model_dir /data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv3undirectedjoint_0/model \
#     --output_csv validation_metrics.csv

import os
import re
import pandas as pd
import argparse

# Define the path to your log files and model files
# These will be overridden by command-line arguments
default_log_dir = '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv3undirectedjoint_0/model'
default_model_dir = '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv3undirectedjoint_0/model'  # Assuming models are in the same directory

# Define the output CSV file name
default_output_csv = 'validation_metrics.csv'

# Define possible modes and their applicable metrics
modes_metrics = {
    '_norecon_class': ['max_epochs', 'max_validation_dice', 'max_validation_dice_epoch',
                       'max_validation_dice_epoch_filename', 'best_model_filename'],
    '_recon_noclass': ['max_epochs', 'min_recon_error', 'min_recon_error_epoch',
                       'min_recon_error_filename', 'best_model_filename'],
    '_recon_class': ['max_epochs', 'max_validation_dice', 'max_validation_dice_epoch',
                     'max_validation_dice_epoch_filename', 'min_recon_error', 'min_recon_error_epoch',
                     'min_recon_error_filename', 'best_model_filename']
}

# Create MultiIndex for columns
tuples = []
# First three columns are surf_type, surf_hemi, GAT Layers with no sub-columns
tuples.extend([('', 'surf_type'), ('', 'surf_hemi'), ('', 'GAT Layers')])

# For each mode, add the applicable metrics
for mode, metrics in modes_metrics.items():
    for metric in metrics:
        tuples.append((metric, mode))

# Create MultiIndex
index = pd.MultiIndex.from_tuples(tuples, names=['Metric', 'Mode'])

# Function to parse model filenames
def parse_model_filenames(model_files):
    """
    Parses model filenames and returns a dictionary mapping from (random_number, epoch) to model filename.
    """
    model_dict = {}
    for filename in model_files:
        pattern = r'^model_.*?_(\d+)epochs_.*?_(\d{6})\.pt$'
        match = re.match(pattern, filename)
        if match:
            epoch = int(match.group(1))
            random_number = match.group(2)
            key = (random_number, epoch)
            model_dict[key] = filename
    return model_dict

# Function to parse log filenames
def parse_log_filename(filename):
    """
    Parses the log filename to extract parameters.
    Example filename:
    model_gm_hcp_lh_vc_v3_csrvc_layers12_sf0.1_heads1.log
    """
    pattern = r'^model_(wm|gm)_hcp_(lh|rh)_.*?_layers(\d+).*?_(norecon_class|recon_noclass|recon_class)_(\d{6})_heads\d+\.log$'
    match = re.match(pattern, filename)
    if match:
        surf_type = match.group(1)
        surf_hemi = match.group(2)
        gat_layers = int(match.group(3))
        mode = '_' + match.group(4)
        random_number = match.group(5)
        return surf_type, surf_hemi, gat_layers, mode, random_number
    else:
        return None

# Function to parse log contents
def parse_log_file(filepath, mode, max_epoch):
    """
    Parses the log file to extract validation metrics up to max_epoch.
    Returns a dictionary with extracted metrics and a flag indicating if data was found.
    """
    metrics_extracted = {}
    data_found = False  # Flag to check if any data was found

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
                data_found = True
                epoch = int(epoch_match.group(1))
                
                # Skip epochs beyond max_epoch
                if epoch > max_epoch:
                    continue

                if epoch > max_epochs:
                    max_epochs = epoch

                # Check for dice validation error (only if classification is involved)
                if '_class' in mode:
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

                # Check for reconstruction validation error (only if reconstruction is involved)
                if 'recon_' in mode or mode == '_recon_class':
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

    # Assign extracted metrics based on mode
    if max_epochs > 0:
        metrics_extracted['max_epochs'] = max_epochs
    else:
        metrics_extracted['max_epochs'] = 'No data found in log file'

    if '_class' in mode:
        if max_dice != -float('inf'):
            metrics_extracted['max_validation_dice'] = max_dice
            metrics_extracted['max_validation_dice_epoch'] = max_dice_epoch
        else:
            metrics_extracted['max_validation_dice'] = 'No dice data found'
            metrics_extracted['max_validation_dice_epoch'] = 'No dice epoch data found'

    if 'recon_' in mode or mode == '_recon_class':
        if min_recon_error != float('inf'):
            metrics_extracted['min_recon_error'] = min_recon_error
            metrics_extracted['min_recon_error_epoch'] = min_recon_error_epoch
        else:
            metrics_extracted['min_recon_error'] = 'No recon error data found'
            metrics_extracted['min_recon_error_epoch'] = 'No recon error epoch data found'

    return metrics_extracted, data_found

# Function to save ground truth surfaces
def save_ground_truth_surf(v_gt, f_gt, save_path_fs, data_name='hcp'):
    """
    Saves the ground truth surface using nibabel.
    """
    verts = v_gt.squeeze()
    faces = f_gt.squeeze().astype(np.int32)

    assert not np.isnan(verts.max()), "The value is NaN in ground truth vertices."
    assert not np.isnan(verts.min()), "The value is NaN in ground truth vertices."

    # Save geometry
    nib.freesurfer.io.write_geometry(save_path_fs + '.surf', verts, faces)

# Function to save predicted surfaces with optional annotations
def save_mesh_with_annotations(verts, faces, labels=None, ctab=None, save_path_fs=None, data_name='hcp', epoch_info=None):
    """
    Save the mesh with annotations using nibabel.

    Parameters:
    - verts: NumPy array of vertices.
    - faces: NumPy array of faces.
    - labels: (Optional) NumPy array of labels.
    - ctab: (Optional) Color table for annotations.
    - save_path_fs: Path to save the FreeSurfer files.
    - data_name: Name of the dataset (e.g., 'hcp', 'adni').
    - epoch_info: (Optional) String containing epoch information to include in file names.
    """
    if save_path_fs is None:
        raise ValueError("save_path_fs must be provided to save the mesh.")

    verts = verts.squeeze()
    faces = faces.squeeze().astype(np.int32)

    assert not np.isnan(verts.max()), "The value is NaN in vertices."
    assert not np.isnan(verts.min()), "The value is NaN in vertices."

    if labels is not None:
        labels = labels.squeeze().astype(np.int32)
        if isinstance(ctab, torch.Tensor):
            ctab = ctab.numpy()
        ctab = ctab.astype(np.int32)
        assert ctab.shape[1] == 5, "ctab should have 5 columns for RGBA and region labels."

    # Modify save paths to include epoch information
    if epoch_info is not None:
        surf_path = f"{save_path_fs}_epoch{epoch_info}.surf"
        annot_path = f"{save_path_fs}_epoch{epoch_info}.annot"
    else:
        surf_path = f"{save_path_fs}.surf"
        annot_path = f"{save_path_fs}.annot"

    # Save geometry
    nib.freesurfer.io.write_geometry(surf_path, verts, faces)

    # Save annotations if labels are provided
    if labels is not None and ctab is not None:
        nib.freesurfer.io.write_annot(annot_path, labels, ctab, decode_names(), fill_ctab=False)

def main(args):
    # Create MultiIndex for columns
    tuples = []
    # First three columns are surf_type, surf_hemi, GAT Layers with no sub-columns
    tuples.extend([('', 'surf_type'), ('', 'surf_hemi'), ('', 'GAT Layers')])

    # For each mode, add the applicable metrics
    for mode, metrics in modes_metrics.items():
        for metric in metrics:
            tuples.append((metric, mode))

    # Create MultiIndex
    index = pd.MultiIndex.from_tuples(tuples, names=['Metric', 'Mode'])

    # Initialize an empty list to collect rows
    rows_list = []

    # Define possible values based on provided CSV data
    surf_types = ['wm', 'gm']
    surf_hemis = ['lh', 'rh']
    gat_layers = [4, 8]#, 12]

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

    # Function to extract max_epoch per mode
    def get_max_epoch_for_mode(mode, max_epochs_dict):
        """
        Retrieves the max_epoch value for a given mode from the max_epochs_dict.
        """
        return max_epochs_dict.get(mode, 60)  # Default to 60 if not specified

    # Function to decode region names (placeholder)
    def decode_names():
        """
        Placeholder function to decode region names.
        Replace with actual implementation as needed.
        """
        return []

    # Function to map best epoch to model filename
    def get_best_model_filename(model_dict, random_number, best_epoch):
        """
        Retrieves the best model filename based on random_number and best_epoch.
        """
        key = (random_number, best_epoch)
        best_model_filename = model_dict.get(key)
        if best_model_filename:
            return best_model_filename
        else:
            # Try to find a model file with the closest epoch less than or equal to the best epoch
            candidate_epochs = [epoch for (rn, epoch) in model_dict.keys() if rn == random_number and epoch <= best_epoch]
            if candidate_epochs:
                closest_epoch = max(candidate_epochs)
                model_key = (random_number, closest_epoch)
                best_model_filename = model_dict.get(model_key)
                if closest_epoch != best_epoch and best_model_filename:
                    return f"{best_model_filename} (closest to epoch {best_epoch})"
                elif best_model_filename:
                    return best_model_filename
            return 'Model file not found for best epoch'

    # Parse model filenames
    model_files = [f for f in os.listdir(args.model_dir) if f.endswith('.pt')]
    model_dict = parse_model_filenames(model_files)

    # Prepare per-mode max_epoch mapping
    max_epochs_dict = {
        '_norecon_class': args.max_epoch_norecon_class,
        '_recon_noclass': args.max_epoch_recon_noclass,
        '_recon_class': args.max_epoch_recon_class
    }

    # Process each row in the DataFrame
    for idx, row in df.iterrows():
        surf_type = row[('', 'surf_type')]
        surf_hemi = row[('', 'surf_hemi')]
        gat_layers = int(row[('', 'GAT Layers')])

        for mode in modes_metrics.keys():
            # Construct expected log filename pattern
            pattern = rf'^model_{surf_type}_hcp_{surf_hemi}_.*?_layers{gat_layers}.*?{mode}_(\d{{6}})_heads\d+\.log$'
            matching_logs = []

            # Search for matching log files
            for log_file in os.listdir(args.log_dir):
                match = re.match(pattern, log_file)
                if match:
                    random_number = match.group(1)
                    matching_logs.append((log_file, random_number))

            if not matching_logs:
                # No log file exists for this combination
                reason = 'Log file does not exist'
                # Populate all applicable metrics with the reason
                for metric in modes_metrics[mode]:
                    df.at[idx, (metric, mode)] = reason
                continue  # Move to the next mode

            # Use the first matching log file (assuming there should only be one)
            log_file, random_number = matching_logs[0]
            log_path = os.path.join(args.log_dir, log_file)

            # Get the max_epoch for the current mode
            current_max_epoch = get_max_epoch_for_mode(mode, max_epochs_dict)

            # Parse the log file with epoch threshold
            metrics, data_found = parse_log_file(log_path, mode, current_max_epoch)

            # Update the DataFrame with extracted metrics
            for metric, value in metrics.items():
                if value is not None and (metric in modes_metrics[mode]):
                    df.at[idx, (metric, mode)] = value

            # Include the log filename in the appropriate metric
            if 'max_validation_dice_epoch_filename' in modes_metrics[mode]:
                df.at[idx, ('max_validation_dice_epoch_filename', mode)] = log_file
            if 'min_recon_error_filename' in modes_metrics[mode]:
                df.at[idx, ('min_recon_error_filename', mode)] = log_file

            # Find the best model filename corresponding to the best epoch and random number
            best_epoch = None
            if 'max_validation_dice_epoch' in metrics and '_class' in mode:
                best_epoch = metrics['max_validation_dice_epoch']
            elif 'min_recon_error_epoch' in metrics and ('recon_' in mode or mode == '_recon_class'):
                best_epoch = metrics['min_recon_error_epoch']

            if best_epoch is not None and random_number is not None:
                best_model_filename = get_best_model_filename(model_dict, random_number, best_epoch)
                df.at[idx, ('best_model_filename', mode)] = best_model_filename
            else:
                df.at[idx, ('best_model_filename', mode)] = 'Best epoch not found'

            # If no data was found in the log file, provide a reason
            if not data_found:
                reason = 'No matching data found in log file'
                df.at[idx, ('max_epochs', mode)] = reason
                for metric in modes_metrics[mode]:
                    if metric != 'max_epochs':
                        df.at[idx, (metric, mode)] = reason

    # Replace any remaining NaN values with a default reason
    df.fillna('Data not available', inplace=True)

    # Save the DataFrame to CSV with two-level headers
    df.to_csv(args.output_csv, index=False)

    print(f"CSV file '{args.output_csv}' has been created successfully in the current directory.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract Validation Metrics with Epoch Thresholds")

    # Add command-line arguments
    parser.add_argument('--max_epoch_norecon_class', type=int, default=60, help="Max epoch for _norecon_class mode")
    parser.add_argument('--max_epoch_recon_noclass', type=int, default=60, help="Max epoch for _recon_noclass mode")
    parser.add_argument('--max_epoch_recon_class', type=int, default=60, help="Max epoch for _recon_class mode")
    
    parser.add_argument('--log_dir', type=str, default=default_log_dir, help="Directory containing log files")
    parser.add_argument('--model_dir', type=str, default=default_model_dir, help="Directory containing model files")
    parser.add_argument('--output_csv', type=str, default=default_output_csv, help="Output CSV filename")

    args = parser.parse_args()
    main(args)
