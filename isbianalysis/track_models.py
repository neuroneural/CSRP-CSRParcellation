import os
import re
import pandas as pd
import argparse
import logging

# Configure logging
logging.basicConfig(
    filename='track_models.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Define the default paths (these can be overridden by command-line arguments)
default_log_dir = '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv4_0/model'
default_model_dir = '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv4_0/model'
default_output_csv = 'validation_metrics.csv'

# Define possible modes and their applicable metrics
modes_metrics = {
    '_norecon_class': [
        'max_epochs',
        'chamfer validation error',
        'chamfer validation error_epoch',
        'chamfer validation error_epoch_filename',
        'best_model_filename'
    ],
    '_recon_noclass': [
        'max_epochs',
        'chamfer validation error',
        'chamfer validation error_epoch',
        'chamfer validation error_epoch_filename',
        'best_model_filename'
    ],
    '_recon_class': [
        'max_epochs',
        'in_dist_dice validation error',
        'in_dist_dice validation error_epoch',
        'in_dist_dice validation error_epoch_filename',
        'best_model_filename'
    ]
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

def parse_model_filenames(model_files):
    """
    Parses model filenames and returns a dictionary mapping from (random_number, epoch) to model filename.
    """
    model_dict = {}
    # Regex to match the model filename pattern
    pattern = r'^model_(wm|gm)_hcp_(lh|rh)_vc_v4_csrvc_layers(\d+)_sf0\.1_heads\d+_' \
              r'(\d+)epochs_euler_(norecon_class|recon_noclass|recon_class)_de0\.1_' \
              r'(\d{6})(?:_final)?\.pt$'
    for filename in model_files:
        match = re.match(pattern, filename)
        if match:
            surf_type = match.group(1)
            surf_hemi = match.group(2)
            gat_layers = int(match.group(3))
            epochs = int(match.group(4))  # Ensure epoch is int
            mode = '_' + match.group(5)
            random_number = match.group(6)
            key = (random_number, epochs)
            model_dict[key] = filename
            logging.info(f"Parsed model file '{filename}': surf_type={surf_type}, surf_hemi={surf_hemi}, "
                         f"gat_layers={gat_layers}, epochs={epochs}, mode={mode}, random_number={random_number}")
        else:
            logging.warning(f"Failed to parse model filename: {filename}")
    return model_dict

def parse_log_filename(filename):
    """
    Parses the log filename to extract parameters.
    """
    pattern = r'^model_(wm|gm)_hcp_(lh|rh)_vc_v4_csrvc_layers(\d+)_sf0\.1_euler_' \
              r'(norecon_class|recon_noclass|recon_class)_de0\.1_' \
              r'(\d{6})_heads\d+\.log$'
    match = re.match(pattern, filename)
    if match:
        surf_type = match.group(1)
        surf_hemi = match.group(2)
        gat_layers = int(match.group(3))
        mode = '_' + match.group(4)
        random_number = match.group(5)
        logging.info(f"Parsed log file '{filename}': surf_type={surf_type}, surf_hemi={surf_hemi}, "
                     f"gat_layers={gat_layers}, mode={mode}, random_number={random_number}")
        return surf_type, surf_hemi, gat_layers, mode, random_number
    else:
        logging.warning(f"Failed to parse log filename: {filename}")
        return None

def parse_log_file(filepath, mode, max_epoch):
    """
    Parses the log file to extract validation metrics up to max_epoch.
    Returns a dictionary with extracted metrics and a flag indicating if data was found.
    """
    metrics_extracted = {}
    data_found = False  # Flag to check if any data was found

    # Initialize variables based on mode
    if '_class' in mode:
        # For classification modes
        target_metric = 'in_dist_dice validation error'
        best_value = -float('inf')  # We want to maximize
        best_epoch_value = None
    else:
        # For non-classification modes
        target_metric = 'chamfer validation error'
        best_value = float('inf')  # We want to minimize
        best_epoch_value = None

    try:
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

                    # Update max_epochs
                    if epoch > metrics_extracted.get('max_epochs', 0):
                        metrics_extracted['max_epochs'] = epoch

                    # Extract target metric
                    metric_match = re.search(rf'{re.escape(target_metric)}:([^\s]+)', line)
                    if metric_match:
                        metric_value_str = metric_match.group(1)
                        if metric_value_str.lower() == 'nan':
                            continue  # Skip NaN values
                        try:
                            metric_value = float(metric_value_str)
                            if '_class' in mode:
                                # Maximize the metric
                                if metric_value > best_value:
                                    best_value = metric_value
                                    best_epoch_value = epoch
                            else:
                                # Minimize the metric
                                if metric_value < best_value:
                                    best_value = metric_value
                                    best_epoch_value = epoch
                        except ValueError:
                            logging.warning(f"Invalid value for '{target_metric}' in file '{filepath}' at epoch {epoch}")
    except Exception as e:
        logging.error(f"Error reading log file '{filepath}': {e}")
        return metrics_extracted, data_found

    # Assign extracted metrics based on mode
    if best_epoch_value is not None:
        metrics_extracted[target_metric] = best_value
        metrics_extracted[f'{target_metric}_epoch'] = best_epoch_value

    return metrics_extracted, data_found

def get_best_model_filename(model_dict, random_number, best_epoch):
    """
    Retrieves the best model filename based on random_number and best_epoch.
    """
    if not isinstance(best_epoch, int):
        return 'Best epoch not valid'

    key = (random_number, best_epoch)
    best_model_filename = model_dict.get(key)
    if best_model_filename:
        return best_model_filename
    else:
        # Try to find a model file with the closest epoch less than or equal to the best epoch
        candidate_epochs = [epoch for (rn, epoch) in model_dict.keys() if rn == random_number and isinstance(epoch, int) and epoch <= best_epoch]
        if candidate_epochs:
            closest_epoch = max(candidate_epochs)
            model_key = (random_number, closest_epoch)
            best_model_filename = model_dict.get(model_key)
            if closest_epoch != best_epoch and best_model_filename:
                return f"{best_model_filename} (closest to epoch {best_epoch})"
            elif best_model_filename:
                return best_model_filename
        return 'Model file not found for best epoch'

def main(args):
    # Verify and list log files
    logging.info(f"Listing files in log_dir: {args.log_dir}")
    try:
        log_files = [f for f in os.listdir(args.log_dir) if f.endswith('.log')]
        logging.info(f"Found {len(log_files)} log files.")
    except Exception as e:
        logging.error(f"Error accessing log_dir: {e}")
        return

    # Parse all log files and organize them
    parsed_logs = {}
    for log_file in log_files:
        parsed = parse_log_filename(log_file)
        if parsed:
            surf_type, surf_hemi, gat_layers, mode, random_number = parsed
            key = (surf_type, surf_hemi, gat_layers, mode)
            if key not in parsed_logs:
                parsed_logs[key] = []
            parsed_logs[key].append(random_number)
        else:
            logging.warning(f"Failed to parse log filename: {log_file}")

    # Verify and list model files
    logging.info(f"Listing files in model_dir: {args.model_dir}")
    try:
        model_files = [f for f in os.listdir(args.model_dir) if f.endswith('.pt')]
        logging.info(f"Found {len(model_files)} model files.")
    except Exception as e:
        logging.error(f"Error accessing model_dir: {e}")
        return

    # Parse model filenames
    model_dict = parse_model_filenames(model_files)

    # Prepare per-mode max_epoch mapping
    max_epochs_dict = {
        '_norecon_class': args.max_epoch_norecon_class,
        '_recon_noclass': args.max_epoch_recon_noclass,
        '_recon_class': args.max_epoch_recon_class
    }

    # Initialize an empty list to collect rows
    rows_list = []

    # Define possible values based on provided CSV data
    surf_types = ['wm', 'gm']
    surf_hemis = ['lh', 'rh']
    gat_layers = [8]  # Adjust as necessary

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

    # Create DataFrame from the list of rows with dtype=object to handle mixed data types
    df = pd.DataFrame(rows_list, columns=index, dtype=object)

    # Process each row in the DataFrame
    for idx, row in df.iterrows():
        surf_type = row[('', 'surf_type')]
        surf_hemi = row[('', 'surf_hemi')]
        gat_layers = int(row[('', 'GAT Layers')])

        for mode in modes_metrics.keys():
            key = (surf_type, surf_hemi, gat_layers, mode)
            random_numbers = parsed_logs.get(key)

            if not random_numbers:
                # No log files exist for this combination
                reason = 'Log file does not exist'
                for metric in modes_metrics[mode]:
                    df.at[idx, (metric, mode)] = reason
                logging.info(f"No log files for combination: surf_type={surf_type}, surf_hemi={surf_hemi}, "
                             f"gat_layers={gat_layers}, mode={mode}")
                continue  # Move to the next mode

            # Initialize variables to track the best metrics across all random_numbers
            best_metric_value = None
            best_epoch = None
            best_random_number = None
            best_metrics = {}
            data_found_any = False  # Flag to check if any data was found in any log

            for random_number in random_numbers:
                # Construct the exact log filename
                log_filename = f"model_{surf_type}_hcp_{surf_hemi}_vc_v4_csrvc_layers{gat_layers}_" \
                               f"sf0.1_euler{mode}_de0.1_{random_number}_heads1.log"
                log_path = os.path.join(args.log_dir, log_filename)

                # Check if the log file actually exists
                if not os.path.isfile(log_path):
                    logging.warning(f"Log file does not exist: {log_path}")
                    continue

                # Define target_metric based on mode
                if '_class' in mode:
                    target_metric = 'in_dist_dice validation error'
                    is_classification = True
                else:
                    target_metric = 'chamfer validation error'
                    is_classification = False

                # Parse the log file
                logging.info(f"Parsing log file: {log_path}")
                metrics, data_found = parse_log_file(log_path, mode, max_epochs_dict.get(mode, 60))

                if data_found:
                    data_found_any = True

                # Check if metrics were extracted
                if target_metric in metrics and f'{target_metric}_epoch' in metrics:
                    current_metric_value = metrics[target_metric]
                    current_epoch = metrics[f'{target_metric}_epoch']

                    # Compare with the best metric value
                    if best_metric_value is None:
                        best_metric_value = current_metric_value
                        best_epoch = current_epoch
                        best_random_number = random_number
                        best_metrics = metrics.copy()
                        best_log_filename = log_filename
                    else:
                        if is_classification:
                            # For classification, maximize the metric
                            if current_metric_value > best_metric_value:
                                best_metric_value = current_metric_value
                                best_epoch = current_epoch
                                best_random_number = random_number
                                best_metrics = metrics.copy()
                                best_log_filename = log_filename
                        else:
                            # For regression, minimize the metric
                            if current_metric_value < best_metric_value:
                                best_metric_value = current_metric_value
                                best_epoch = current_epoch
                                best_random_number = random_number
                                best_metrics = metrics.copy()
                                best_log_filename = log_filename

            if best_metric_value is not None:
                # Update the DataFrame with the best metrics
                for metric, value in best_metrics.items():
                    if value is not None and (metric in modes_metrics[mode]):
                        df.at[idx, (metric, mode)] = value

                # Include the log filename in the appropriate metric
                if f'{target_metric}_epoch_filename' in modes_metrics[mode]:
                    df.at[idx, (f'{target_metric}_epoch_filename', mode)] = best_log_filename

                # Find the best model filename corresponding to the best epoch and random number
                if isinstance(best_epoch, int):
                    best_model_filename = get_best_model_filename(model_dict, best_random_number, best_epoch)
                    df.at[idx, ('best_model_filename', mode)] = best_model_filename
                    logging.info(f"Best model for mode {mode}: {best_model_filename}")
                else:
                    df.at[idx, ('best_model_filename', mode)] = 'Best epoch not found'
                    logging.info(f"Best epoch is not valid for combination: surf_type={surf_type}, surf_hemi={surf_hemi}, "
                                 f"gat_layers={gat_layers}, mode={mode}")
            else:
                # No valid metrics found in any log file
                reason = 'No valid data found in any log file'
                for metric in modes_metrics[mode]:
                    df.at[idx, (metric, mode)] = reason
                logging.warning(f"No valid data found for combination: surf_type={surf_type}, surf_hemi={surf_hemi}, "
                                f"gat_layers={gat_layers}, mode={mode}")

    # Replace any remaining NaN values with a default reason
    df.fillna('Data not available', inplace=True)

    # Save the DataFrame to CSV with two-level headers
    df.to_csv(args.output_csv, index=False)

    logging.info(f"CSV file '{args.output_csv}' has been created successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract Validation Metrics with Epoch Thresholds")

    # Add command-line arguments
    parser.add_argument('--max_epoch_norecon_class', type=int, default=100, help="Max epoch for _norecon_class mode")
    parser.add_argument('--max_epoch_recon_noclass', type=int, default=100, help="Max epoch for _recon_noclass mode")
    parser.add_argument('--max_epoch_recon_class', type=int, default=100, help="Max epoch for _recon_class mode")

    parser.add_argument('--log_dir', type=str, default=default_log_dir, help="Directory containing log files")
    parser.add_argument('--model_dir', type=str, default=default_model_dir, help="Directory containing model files")
    parser.add_argument('--output_csv', type=str, default=default_output_csv, help="Output CSV filename")

    args = parser.parse_args()
    main(args)
