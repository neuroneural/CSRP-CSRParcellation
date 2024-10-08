import pandas as pd
import numpy as np
from scipy import stats
import os

def process_and_summarize_data(cortexode_csv, csrp_separate_csv, csrp_unified_csv, output_csv):
    # Read the CSV files
    try:
        cortexode_df = pd.read_csv(cortexode_csv)
        csrp_separate_df = pd.read_csv(csrp_separate_csv)
        csrp_unified_df = pd.read_csv(csrp_unified_csv)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    # Add 'Case' column to cortexode_df
    cortexode_df['Case'] = 'baseline'  # or 'unified', depending on your preference

    # Combine the DataFrames
    all_data = pd.concat([cortexode_df, csrp_separate_df, csrp_unified_df], ignore_index=True)

    # Ensure 'GNNLayers' is numeric
    all_data['GNNLayers'] = pd.to_numeric(all_data['GNNLayers'], errors='coerce')

    # Ensure 'Score' is numeric
    all_data['Score'] = pd.to_numeric(all_data['Score'], errors='coerce')

    # Remove rows with missing 'Score' or 'GNNLayers'
    all_data = all_data.dropna(subset=['Score', 'GNNLayers'])

    # Grouping columns
    grouping_cols = ['Framework', 'Case', 'surf_type', 'GNNLayers', 'Metric']

    # Prepare an empty list to collect summary data
    summary_data = []

    # Get unique combinations for grouping
    unique_groups = all_data[grouping_cols].drop_duplicates()

    # For each unique group, calculate statistics
    for _, group_info in unique_groups.iterrows():
        framework = group_info['Framework']
        case = group_info['Case']
        surf_type = group_info['surf_type']
        gnn_layers = group_info['GNNLayers']
        metric = group_info['Metric']

        # Filter data for the current group
        group_data = all_data[
            (all_data['Framework'] == framework) &
            (all_data['Case'] == case) &
            (all_data['surf_type'] == surf_type) &
            (all_data['GNNLayers'] == gnn_layers) &
            (all_data['Metric'] == metric)
        ]

        # Calculate mean and std
        mean_score = group_data['Score'].mean()
        std_score = group_data['Score'].std()

        # Prepare a summary row
        summary_row = {
            'Framework': framework,
            'Case': case,
            'surf_type': surf_type,
            'GNNLayers': gnn_layers,
            'Metric': metric,
            'Average': mean_score,
            'StdDev': std_score
        }

        summary_data.append(summary_row)

    # Convert summary data to DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Now, compute p-values between groups
    # Prepare a DataFrame to collect p-values
    p_values = []

    # Comparison 1: csrp unified vs csrp separate
    for metric in ['Chamfer Distance', 'Hausdorff Distance', 'Self-Intersections (SIF)']:
        for surf_type in all_data['surf_type'].unique():
            for gnn_layers in all_data['GNNLayers'].unique():
                # Get data for csrp unified
                csrp_unified_data = all_data[
                    (all_data['Framework'] == 'csrp') &
                    (all_data['Case'] == 'unified') &
                    (all_data['surf_type'] == surf_type) &
                    (all_data['GNNLayers'] == gnn_layers) &
                    (all_data['Metric'] == metric)
                ]['Score']

                # Get data for csrp separate
                csrp_separate_data = all_data[
                    (all_data['Framework'] == 'csrp') &
                    (all_data['Case'] == 'separate') &
                    (all_data['surf_type'] == surf_type) &
                    (all_data['GNNLayers'] == gnn_layers) &
                    (all_data['Metric'] == metric)
                ]['Score']

                if not csrp_unified_data.empty and not csrp_separate_data.empty:
                    # Compute p-value
                    t_stat, p_value = stats.ttest_ind(
                        csrp_unified_data, csrp_separate_data, equal_var=False, nan_policy='omit'
                    )
                    # Append p-value for csrp unified
                    p_values.append({
                        'Framework': 'csrp',
                        'Case': 'unified',
                        'surf_type': surf_type,
                        'GNNLayers': gnn_layers,
                        'Metric': metric,
                        'P-value': p_value
                    })
                    # Append p-value for csrp separate
                    p_values.append({
                        'Framework': 'csrp',
                        'Case': 'separate',
                        'surf_type': surf_type,
                        'GNNLayers': gnn_layers,
                        'Metric': metric,
                        'P-value': p_value
                    })

    # Comparison 2: csrp (each case) vs cortexode
    for metric in ['Chamfer Distance', 'Hausdorff Distance', 'Self-Intersections (SIF)']:
        for surf_type in all_data['surf_type'].unique():
            for gnn_layers in all_data['GNNLayers'].unique():
                # Get data for cortexode
                cortexode_data = all_data[
                    (all_data['Framework'] == 'cortexode') &
                    (all_data['Case'] == 'baseline') &
                    (all_data['surf_type'] == surf_type) &
                    (all_data['GNNLayers'] == gnn_layers) &
                    (all_data['Metric'] == metric)
                ]['Score']

                for case in ['unified', 'separate']:
                    # Get data for csrp case
                    csrp_case_data = all_data[
                        (all_data['Framework'] == 'csrp') &
                        (all_data['Case'] == case) &
                        (all_data['surf_type'] == surf_type) &
                        (all_data['GNNLayers'] == gnn_layers) &
                        (all_data['Metric'] == metric)
                    ]['Score']

                    if not csrp_case_data.empty and not cortexode_data.empty:
                        # Compute p-value
                        t_stat, p_value = stats.ttest_ind(
                            csrp_case_data, cortexode_data, equal_var=False, nan_policy='omit'
                        )
                        # Append p-value for csrp case
                        p_values.append({
                            'Framework': 'csrp',
                            'Case': case,
                            'surf_type': surf_type,
                            'GNNLayers': gnn_layers,
                            'Metric': metric,
                            'P-value': p_value
                        })
                        # Append p-value for cortexode
                        p_values.append({
                            'Framework': 'cortexode',
                            'Case': 'baseline',
                            'surf_type': surf_type,
                            'GNNLayers': gnn_layers,
                            'Metric': metric,
                            'P-value': p_value
                        })

    # Convert p_values to DataFrame
    p_values_df = pd.DataFrame(p_values)

    # Merge summary_df and p_values_df on all keys
    summary_df = summary_df.merge(
        p_values_df,
        how='left',
        on=['Framework', 'Case', 'surf_type', 'GNNLayers', 'Metric']
    )

    # Reorder columns
    summary_df = summary_df[
        ['Framework', 'Case', 'surf_type', 'GNNLayers', 'Metric', 'Average', 'StdDev', 'P-value']
    ]

    # Save summary to CSV
    summary_df.to_csv(output_csv, index=False)
    print(f"Summary table saved to {output_csv}")

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process and summarize metrics from CSV files.')
    parser.add_argument('--cortexode_csv', type=str, default='cortexode_measures.csv', help='Path to cortexode CSV file')
    parser.add_argument('--csrp_separate_csv', type=str, default='csrp_measures-400-100-100-a.csv', help='Path to csrp separate CSV file')
    parser.add_argument('--csrp_unified_csv', type=str, default='csrp_measures-400-100-100-b.csv', help='Path to csrp unified CSV file')
    parser.add_argument('--output_csv', type=str, default='deformationtable.csv', help='Path to output summary CSV file')


    args = parser.parse_args()

    process_and_summarize_data(
        cortexode_csv=args.cortexode_csv,
        csrp_separate_csv=args.csrp_separate_csv,
        csrp_unified_csv=args.csrp_unified_csv,
        output_csv=args.output_csv
    )
