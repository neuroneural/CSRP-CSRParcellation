import pandas as pd
import numpy as np

def compute_macro_dice(df):
    # Exclude label == 4 and Metric == 'Dice'
    df_filtered = df[(df['Label'] != 4) & (df['Metric'] == 'Dice')]

    # Group by required fields and compute mean and std
    grouped = df_filtered.groupby(['Framework', 'surf_type', 'GNNLayers', 'Metric'])
    mean_std = grouped['Score'].agg(['mean', 'std']).reset_index()

    return mean_std

def compute_macro_dice_per_subject(df):
    # Exclude label == 4 and Metric == 'Dice'
    df_filtered = df[(df['Label'] != 4) & (df['Metric'] == 'Dice')]

    # Group by Subject ID and compute mean Dice score
    grouped = df_filtered.groupby(['Subject ID', 'Framework', 'surf_type', 'GNNLayers', 'Metric'])
    mean_dice = grouped['Score'].mean().reset_index()

    return mean_dice

def get_top5_subjects_by_macro_dice(df):
    # Compute macro Dice per subject
    mean_dice = compute_macro_dice_per_subject(df)

    # Sort by 'Score' in descending order
    mean_dice_sorted = mean_dice.sort_values(by='Score', ascending=False)

    # Get top 5 subjects
    top5_subjects = mean_dice_sorted.head(5).reset_index(drop=True)

    return top5_subjects

def process_csv_files(file_paths):
    all_mean_std = []
    all_top5 = []

    for file_path in file_paths:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Ensure 'Label' is numeric (in case it's read as string)
        df['Label'] = pd.to_numeric(df['Label'], errors='coerce')

        # Compute Macro Dice
        mean_std = compute_macro_dice(df)
        mean_std['CSV_File'] = file_path  # Add filename for reference
        all_mean_std.append(mean_std)

        # Get Top 5 Subjects by Macro Dice
        top5_subjects = get_top5_subjects_by_macro_dice(df)
        top5_subjects['CSV_File'] = file_path  # Add filename for reference
        all_top5.append(top5_subjects)

    # Combine results from all files
    combined_mean_std = pd.concat(all_mean_std, ignore_index=True)
    combined_top5 = pd.concat(all_top5, ignore_index=True)

    return combined_mean_std, combined_top5

if __name__ == "__main__":
    # List of CSV file paths
    csv_files = [
        'csrp_dice-100-100-100a.csv',
        'csrp_dice-400-100-100a.csv',
        'csrp_dice-400-100-100b.csv',
        'fastsurfer_dice.csv'# Verify if this is intentional
    ]

    combined_mean_std, combined_top5 = process_csv_files(csv_files)

    # Save the aggregated mean and std to a CSV
    combined_mean_std.to_csv('macro_dice_statistics.csv', index=False)

    # Save the top 5 subjects by macro Dice to a CSV
    combined_top5.to_csv('top5_subjects_by_macro_dice.csv', index=False)

    print("Macro Dice statistics saved to 'macro_dice_statistics.csv'")
    print("Top 5 subjects by macro Dice saved to 'top5_subjects_by_macro_dice.csv'")
