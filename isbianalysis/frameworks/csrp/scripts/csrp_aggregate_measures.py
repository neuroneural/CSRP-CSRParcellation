import os
import glob
import pandas as pd
import numpy as np
from scipy import stats

def process_csv_files(input_directory, output_csv):
    # Initialize a list to hold DataFrames for all files
    all_data = []

    # Define the pattern for CSV files (both 'a' and 'b')
    csv_files = glob.glob(os.path.join(input_directory, '*.csv'))

    # Check if there are any CSV files
    if not csv_files:
        print("No CSV files found in the specified directory.")
        return

    # Process each CSV file
    for file_path in csv_files:
        # Determine group from filename
        filename = os.path.basename(file_path)
        if filename.endswith('a.csv'):
            group = 'separate'
        elif filename.endswith('b.csv'):
            group = 'joint'
        else:
            # Skip files that don't match the pattern
            print(f"Skipping file (does not match pattern): {filename}")
            continue

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Add the 'Group' column
            df['Group'] = group

            # Append to the list
            all_data.append(df)
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
            continue

    # Concatenate all data into a single DataFrame
    data = pd.concat(all_data, ignore_index=True)

    # Ensure required columns are present
    required_columns = ['Group', 'Framework', 'surf_type', 'GNNLayers', 'Metric', 'Score']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return

    # Convert 'GNNLayers' to numeric if it's not already
    data['GNNLayers'] = pd.to_numeric(data['GNNLayers'], errors='coerce')

    # Remove rows with missing 'GNNLayers'
    data = data.dropna(subset=['GNNLayers'])

    # Convert 'Score' to numeric
    data['Score'] = pd.to_numeric(data['Score'], errors='coerce')

    # Remove rows with missing 'Score'
    data = data.dropna(subset=['Score'])

    # Grouping columns
    grouping_cols = ['Framework', 'surf_type', 'GNNLayers', 'Metric']

    # Prepare an empty list to collect summary data
    summary_data = []

    # Get unique combinations of Framework, surf_type, GNNLayers, Metric
    unique_groups = data[grouping_cols].drop_duplicates()

    # For each unique group, calculate statistics
    for _, group_info in unique_groups.iterrows():
        framework = group_info['Framework']
        surf_type = group_info['surf_type']
        gnn_layers = group_info['GNNLayers']
        metric = group_info['Metric']

        # Prepare a dictionary to hold summary for this group
        summary_row = {
            'Framework': framework,
            'surf_type': surf_type,
            'GNNLayers': gnn_layers,
            'Metric': metric
        }

        # Separate data for joint and separate groups
        group_data = data[
            (data['Framework'] == framework) &
            (data['surf_type'] == surf_type) &
            (data['GNNLayers'] == gnn_layers) &
            (data['Metric'] == metric)
        ]

        joint_data = group_data[group_data['Group'] == 'joint']['Score']
        separate_data = group_data[group_data['Group'] == 'separate']['Score']

        # Calculate statistics for joint group
        if not joint_data.empty:
            summary_row['Joint Average'] = joint_data.mean()
            summary_row['Joint StdDev'] = joint_data.std()
        else:
            summary_row['Joint Average'] = np.nan
            summary_row['Joint StdDev'] = np.nan

        # Calculate statistics for separate group
        if not separate_data.empty:
            summary_row['Separate Average'] = separate_data.mean()
            summary_row['Separate StdDev'] = separate_data.std()
        else:
            summary_row['Separate Average'] = np.nan
            summary_row['Separate StdDev'] = np.nan

        # Calculate p-value
        if not joint_data.empty and not separate_data.empty:
            # Use t-test (assuming independent samples)
            t_stat, p_value = stats.ttest_ind(joint_data, separate_data, equal_var=False, nan_policy='omit')
            summary_row['P-value'] = p_value
        else:
            summary_row['P-value'] = np.nan

        # Append the summary row to the list
        summary_data.append(summary_row)

    # Convert the summary data into a DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Reorder columns for clarity
    summary_df = summary_df[
        ['Framework', 'surf_type', 'GNNLayers', 'Metric',
         'Joint Average', 'Joint StdDev',
         'Separate Average', 'Separate StdDev',
         'P-value']
    ]

    # Sort the summary DataFrame
    summary_df = summary_df.sort_values(by=['Framework', 'surf_type', 'GNNLayers', 'Metric'])

    # Save the summary DataFrame to CSV
    summary_df.to_csv(output_csv, index=False)

    print(f"Summary CSV has been saved to: {output_csv}")

# Usage example
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process CSV files to generate summary statistics.')
    parser.add_argument('--input_directory', type=str, required=True, help='Directory containing input CSV files.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to the output summary CSV file.')

    args = parser.parse_args()

    process_csv_files(args.input_directory, args.output_csv)
