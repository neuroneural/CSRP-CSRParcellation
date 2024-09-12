import pandas as pd
import os

# Directories to search for CSV files
base_dirs = [
    '/data/users2/washbee/CortexODE-CSRFusionNet/wacvanalysis/frameworks/csrfparc/assym',
    '/data/users2/washbee/CortexODE-CSRFusionNet/wacvanalysis/frameworks/csrfparc/assymtoassym',
    '/data/users2/washbee/CortexODE-CSRFusionNet/wacvanalysis/frameworks/csrfparc/assymtoassymindist',
    '/data/users2/washbee/CortexODE-CSRFusionNet/wacvanalysis/frameworks/csrfparc/sym',
    '/data/users2/washbee/CortexODE-CSRFusionNet/wacvanalysis/frameworks/csrfparc/symtoassym',
    '/data/users2/washbee/CortexODE-CSRFusionNet/wacvanalysis/frameworks/csrfparc/symtoassymindist'
]

# Initialize an empty DataFrame to store all data
all_data = pd.DataFrame()

# Iterate over base directories and read CSV files
for base_dir in base_dirs:
    csv_path = os.path.join(base_dir, 'csvs', 'grouped_dice_scores.csv')#this file name
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        all_data = pd.concat([all_data, df], ignore_index=True)

# Filter out unwanted columns and rows
all_data = all_data[['gnn_layers', 'surf_hemi', 'surf_type', 'average_dice_score', 'stdev_dice']]

# Replace values in the DataFrame
all_data['surf_hemi'] = all_data['surf_hemi'].replace({'lh': 'left hemisphere', 'rh': 'right hemisphere'})
all_data['surf_type'] = all_data['surf_type'].replace({'gm': 'pial'})

# Create pivot tables for the average dice score and standard deviation
pivot_avg = all_data.pivot_table(index='gnn_layers', columns=['surf_hemi', 'surf_type'], values='average_dice_score')
pivot_stdev = all_data.pivot_table(index='gnn_layers', columns=['surf_hemi', 'surf_type'], values='stdev_dice')

# Save the pivot tables to CSV files
output_csv_path_avg = '../csvs/csrfparcdice_avg.csv'
output_csv_path_stdev = '../csvs/csrfparcdice_stdev.csv'
pivot_avg.to_csv(output_csv_path_avg)
pivot_stdev.to_csv(output_csv_path_stdev)

print("Pivot tables saved to", output_csv_path_avg, "and", output_csv_path_stdev)
