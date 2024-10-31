import pandas as pd

# List of your CSV files corresponding to each conditionisbianalysis/frameworks/freesurfercsrp/csvs/freesurfercsrp_measures_v2.3.csv
filenames = ['../csrp/csvs/csrp_dice_a.csv', '../csrp/csvs/csrp_dice_b.csv', '../cortexode/csvs/cortexode_dice_c.csv', '../freesurfercsrp/csvs/freesurfercsrp_measures_v2.3.csv', '../fastsurfer/csvs/fastsurfer_measures_v2.3.csv']

# Read and concatenate all CSV files into one DataFrame
df_list = []
for filename in filenames:
    df = pd.read_csv(filename)
    df_list.append(df)
df = pd.concat(df_list, ignore_index=True)

# Ensure the 'Score' column is numeric
df['Score'] = pd.to_numeric(df['Score'], errors='coerce')

# First table: Average and standard deviation of Chamfer, Hausdorff, and SIF
metrics1 = ['Chamfer Distance', 'Hausdorff Distance', 'Self-Intersections (SIF)']
first_table_data = df[df['Metric'].isin(metrics1)]
first_table_summary = first_table_data.groupby(['Condition','Metric', 'surf_type'])['Score'].agg(['mean', 'std']).reset_index()

# Save the first table to a CSV file
first_table_summary.to_csv('metrics_summary.csv', index=False)

# Second table: Average and standard deviation of Dice scores
metrics2 = ['Macro Dice']
second_table_data = df[df['Metric'].isin(metrics2)]
second_table_summary = second_table_data.groupby(['Condition','Metric', 'surf_type'])['Score'].agg(['mean', 'std']).reset_index()

# Save the second table to a CSV file
second_table_summary.to_csv('dice_summary.csv', index=False)
