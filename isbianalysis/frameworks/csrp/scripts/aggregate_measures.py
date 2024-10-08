import pandas as pd
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Aggregate CSRFusionNet measures.')
    parser.add_argument('--data_csv', required=True, help='Path to csrf_measures.csv')
    parser.add_argument('--exclusion_csv', required=True, help='Path to lowerstdevs.csv')
    parser.add_argument('--output_csv', required=True, help='Path to output summary CSV')
    args = parser.parse_args()

    # Read the provided CSV data
    df = pd.read_csv(args.data_csv)

    # Read the exclusion data into a DataFrame
    exclusion_df = pd.read_csv(args.exclusion_csv)

    # Filter out subjects with min_stdevfrommean less than -2
    excluded_subjects = exclusion_df[exclusion_df['min_stdevfrommean'] < -2]['subject_id']
    df = df[~df['Subject ID'].isin(excluded_subjects)]

    print('Number of excluded subjects:', len(excluded_subjects))

    # Group by Hemisphere and surf_type
    grouped = df.groupby(['Hemisphere', 'surf_type', 'Metric'])

    # Calculate mean and standard deviation for each group
    results = grouped['Score'].agg(['mean', 'std']).reset_index()

    print('Aggregated Results:', results)

    # Pivot the table for better readability
    pivot_results = results.pivot_table(index=['Hemisphere', 'surf_type'], columns='Metric', values=['mean', 'std'])

    # Flatten the multi-index columns
    pivot_results.columns = ['_'.join(col).strip() for col in pivot_results.columns.values]

    # Save the results to CSV
    pivot_results.to_csv(args.output_csv)

    print(f"Summary saved to {args.output_csv}")

if __name__ == "__main__":
    main()
