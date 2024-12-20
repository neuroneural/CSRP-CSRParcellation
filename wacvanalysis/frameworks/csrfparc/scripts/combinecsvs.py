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

all_data = pd.DataFrame()  # Initialize an empty DataFrame to store all data

# Iterate over each directory, read the CSV, add the new column, and append to the DataFrame
for base_dir in base_dirs:
    csv_path = os.path.join(base_dir, 'csvs', 'grouped_dice_scores.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Extract the last directory name and append '_csrvc'
        last_dir_name = os.path.basename(os.path.normpath(base_dir)) + '_csrvc'
        df['framework'] = last_dir_name  # Add the new column with the modified directory name
        # Ensure 'framework' is the first column
        cols = df.columns.tolist()  # Get a list of all column names
        # Move 'framework' to the first position
        cols = [cols[-1]] + cols[:-1]  # Reorder columns so 'framework' is first
        df = df[cols]  # Apply new column order
        all_data = pd.concat([all_data, df], ignore_index=True)

# Adding simplified categorization columns based on framework naming
all_data['Vertex Mapping'] = all_data['framework'].apply(
    lambda x: 'symmetric' if 'sym' in x else 'asymmetric'
)
all_data['Training Set'] = all_data['framework'].apply(
    lambda x: 'freesurfer' if 'sym' in x else 'cortexode'
)
all_data['Testing Ground Truth'] = all_data['framework'].apply(
    lambda x: 'csrf' if 'toassym' in x and not 'indist' in x else 'cortexode' if 'indist' in x else 'freesurfer'
)

# Reordering columns for a better overview
final_columns = ['Vertex Mapping', 'Training Set', 'Testing Ground Truth', 'framework', 'gnn_layers', 'surf_hemi', 'surf_type', 'average_dice_score', 'stdev_dice', 'count']
all_data = all_data[final_columns]

# Save the organized data to a CSV file
output_csv_path = '/data/users2/washbee/CortexODE-CSRFusionNet/wacvanalysis/frameworks/csrfparc/csvs/organized_dice_scores.csv'
all_data.to_csv(output_csv_path, index=False)

print("Organized dice scores saved to", output_csv_path)
