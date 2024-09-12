import pandas as pd
import numpy as np

# Replace these strings with the paths to your actual CSV files
data_csv_path = '../csvs/cortexode_measures.csv'
exclusion_csv_path = '../../freesurfer/mwremoved/lowerstdevs.csv'

# Read the provided CSV data
df = pd.read_csv(data_csv_path)

# Read the exclusion data into a DataFrame
exclusion_df = pd.read_csv(exclusion_csv_path)

# Filter out subjects with min_stdevfrommean less than -2
excluded_subjects = exclusion_df[exclusion_df['min_stdevfrommean'] < -2]['subject_id']
df = df[~df['Subject ID'].isin(excluded_subjects)]

print('excluded_subjects', len(excluded_subjects))

# Group by Hemisphere and surf_type
grouped = df.groupby(['Hemisphere', 'surf_type', 'Metric'])

# Calculate mean and standard deviation for each group
results = grouped['Score'].agg(['mean', 'std']).reset_index()

print('results', results)
# Pivot the table for better readability
pivot_results = results.pivot_table(index=['Hemisphere', 'surf_type'], columns='Metric', values=['mean', 'std'])

# Flatten the multi-index columns
pivot_results.columns = ['_'.join(col).strip() for col in pivot_results.columns.values]

# Save the results to CSV
output_csv = '../csvs/cortexode_summary.csv'
pivot_results.to_csv(output_csv)

print("Summary saved to", output_csv)
