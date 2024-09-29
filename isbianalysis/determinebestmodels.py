import os
import re
import csv
import pandas as pd
import numpy as np

# Function to parse the filename and extract hyperparameters
def parse_filename(filename):
    # Remove the extension
    filename_no_ext = filename.replace('.log', '')
    # Split the filename into parts
    parts = filename_no_ext.split('_')
    
    # Initialize dictionary to hold hyperparameters
    config = {}
    
    # Surface type
    config['surf_type'] = 'gm' if 'gm' in parts else 'wm' if 'wm' in parts else None
    # Hemisphere
    config['hemisphere'] = 'lh' if 'lh' in parts else 'rh' if 'rh' in parts else None
    # Number of layers
    layers = next((part for part in parts if 'layers' in part), None)
    config['layers'] = int(layers.replace('layers', '')) if layers else None
    # Solver
    config['solver'] = 'euler' if 'euler' in parts else 'rk4' if 'rk4' in parts else None
    # Heads
    heads = next((part for part in parts if 'heads' in part), None)
    config['heads'] = int(heads.replace('heads', '')) if heads else None
    # Version
    version = next((part for part in parts if re.match(r'v\d+', part)), None)
    config['version'] = version if version else None
    # Random number (6-digit number)
    random_number = next((part for part in parts if re.match(r'\d{6}', part)), None)
    config['random_number'] = float(random_number) if random_number else None
    # Model type
    config['model_type'] = 'csrvc' if 'csrvc' in parts else None
    
    # Classification and Reconstruction flags
    # Initialize defaults
    config['classification'] = None
    config['reconstruction'] = None
    
    # Check for specific patterns
    if 'recon_class' in parts:
        config['classification'] = True
        config['reconstruction'] = True
    elif 'norecon_class' in parts:
        config['classification'] = True
        config['reconstruction'] = False
    elif 'recon_noclass' in parts:
        config['classification'] = False
        config['reconstruction'] = True
    elif 'norecon_noclass' in parts:
        config['classification'] = False
        config['reconstruction'] = False
    else:
        # Check for individual 'recon' or 'norecon'
        if 'recon' in parts:
            config['reconstruction'] = True
        elif 'norecon' in parts:
            config['reconstruction'] = False
        else:
            # Default to True if not specified
            config['reconstruction'] = True
            
        # Check for individual 'class' or 'noclass'
        if 'class' in parts:
            config['classification'] = True
        elif 'noclass' in parts:
            config['classification'] = False
        else:
            # Default to True if not specified
            config['classification'] = True
    
    return config

# Function to extract data from log files
def extract_data_from_log(filepath):
    data = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            # Extract epoch number
            epoch_match = re.search(r'epoch:(\d+)', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                # Initialize variables
                recon_error = None
                dice_error = None
                class_error = None

                # Patterns to match numbers, including 'nan' and 'None'
                number_pattern = r'([-+]?\d*\.\d+|\d+|nan|None|NaN)'

                # Check for validation metrics in the same line
                recon_match = re.search(rf'reconstruction validation error:{number_pattern}', line)
                dice_match = re.search(rf'dice validation error:{number_pattern}', line)
                class_match = re.search(rf'classification validation error:{number_pattern}', line)

                def parse_error(match):
                    if match:
                        value = match.group(1)
                        if value.lower() in ['nan', 'none']:
                            return None
                        else:
                            try:
                                return float(value)
                            except ValueError:
                                return None
                    return None

                recon_error = parse_error(recon_match)
                dice_error = parse_error(dice_match)
                class_error = parse_error(class_match)

                # If metrics not found, look ahead in the next few lines
                if recon_error is None or dice_error is None or class_error is None:
                    for j in range(1, 5):
                        if i + j < len(lines):
                            next_line = lines[i + j]
                            if recon_error is None:
                                recon_match = re.search(rf'reconstruction validation error:{number_pattern}', next_line)
                                recon_error = parse_error(recon_match)
                            if dice_error is None:
                                dice_match = re.search(rf'dice validation error:{number_pattern}', next_line)
                                dice_error = parse_error(dice_match)
                            if class_error is None:
                                class_match = re.search(rf'classification validation error:{number_pattern}', next_line)
                                class_error = parse_error(class_match)
                        else:
                            break

                # Append data if any metric is present
                if recon_error is not None or dice_error is not None or class_error is not None:
                    data.append({
                        'epoch': epoch,
                        'recon_error': recon_error,
                        'dice_error': dice_error,
                        'class_error': class_error
                    })
    return data

# Main function to process logs and write CSVs
def main():
    # Directory containing the log files
    log_directory = '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/isbi/isbi_gnnv3undirectedjoint_0/model'  # Update this path
    output_csv = 'aggregated_validation_metrics.csv'
    top_models_csv = 'top_5_models_per_group.csv'
    
    all_data = []
    
    # Get list of log files
    log_files = [f for f in os.listdir(log_directory) if f.endswith('.log')]
    
    for log_file in log_files:
        filepath = os.path.join(log_directory, log_file)
        config = parse_filename(log_file)
        metrics = extract_data_from_log(filepath)
        for metric in metrics:
            data_entry = {**config, **metric}
            data_entry['filename'] = log_file  # Add the filename
            all_data.append(data_entry)
    
    # Convert all_data to DataFrame and remove duplicates
    df = pd.DataFrame(all_data)
    df.drop_duplicates(subset=['filename', 'epoch'], inplace=True)
    
    # Ensure 'classification' and 'reconstruction' are booleans
    df['classification'] = df['classification'].astype(bool)
    df['reconstruction'] = df['reconstruction'].astype(bool)
    
    # Write to CSV
    fieldnames = ['filename', 'surf_type', 'hemisphere', 'layers', 'solver', 'heads', 'version',
                  'random_number', 'model_type', 'classification', 'reconstruction', 'epoch',
                  'recon_error', 'dice_error']
    df.to_csv(output_csv, columns=fieldnames, index=False)
    
    print(f'Data written to {output_csv}')
    
    # Now, find top 5 models per group
    find_top_models(output_csv, top_models_csv)
    print(f'Top models written to {top_models_csv}')

# Function to find top 5 models per group based on relevant validation error
def find_top_models(input_csv, output_csv):
    # Read the CSV with proper converters for booleans
    df = pd.read_csv(
        input_csv,
        dtype={'dice_error': float, 'recon_error': float},
        converters={
            'classification': lambda x: True if x == 'True' else False,
            'reconstruction': lambda x: True if x == 'True' else False
        }
    )
    
    # Fill NaNs with specified values
    df['dice_error'].fillna(-1, inplace=True)
    df['recon_error'].fillna(100, inplace=True)
    
    # Set 'dice_error' to -1 for models where 'classification' is False
    df.loc[df['classification'] == False, 'dice_error'] = -1
    
    # Set 'recon_error' to 100 for models where 'reconstruction' is False
    df.loc[df['reconstruction'] == False, 'recon_error'] = 100
    
    # Group by layers, classification, and reconstruction
    grouped = df.groupby(['layers', 'classification', 'reconstruction'])

    top_models = []

    for name, group in grouped:
        layers, classification, reconstruction = name

        # Initialize group_sorted
        group_sorted = group.copy()

        if classification and reconstruction:
            # Sort by dice_error descending
            group_sorted = group_sorted.sort_values(by='dice_error', ascending=False)
        elif classification and not reconstruction:
            group_sorted = group_sorted.sort_values(by='dice_error', ascending=False)
        elif not classification and reconstruction:
            group_sorted = group_sorted.sort_values(by='recon_error', ascending=True)
        else:
            # Both classification and reconstruction are False
            # There's no meaningful metric to sort by; skip this group
            continue

        # Get top 5 models
        top5 = group_sorted.head(5)
        # Add to list
        top_models.append(top5)

    # Concatenate all top models
    if top_models:
        top_models_df = pd.concat(top_models)
        # Reset index
        top_models_df.reset_index(drop=True, inplace=True)
        # Write to CSV
        output_columns = ['filename', 'surf_type', 'hemisphere', 'layers', 'solver', 'heads',
                          'version', 'random_number', 'model_type', 'classification', 'reconstruction',
                          'epoch', 'recon_error', 'dice_error']
        top_models_df.to_csv(output_csv, columns=output_columns, index=False)
    else:
        print("No models found after grouping.")
        top_models_df = pd.DataFrame()
        top_models_df.to_csv(output_csv, index=False)

if __name__ == '__main__':
    main()
