import pandas as pd

# Step 1: Read the CSV file containing standard deviations and identify subjects to exclude
stdev_df = pd.read_csv('../../../freesurfer/mwremoved/lowerstdevs.csv')
excluded_subjects = stdev_df[stdev_df['min_stdevfrommean'] < -2]['subject_id'].tolist()
print('excluded_subjects',len(excluded_subjects))
# Step 2: Read the CSV file containing dice scores
dice_scores_df = pd.read_csv('../csvs/test_results_vertex_classification_vc.csv')

# Step 3: Exclude the subjects identified in step 1 from the dice scores
filtered_dice_scores_df = dice_scores_df[~dice_scores_df['subject_id'].isin(excluded_subjects)]

print('filtered_dice_scores', len(filtered_dice_scores_df))
# Debugging: Check the counts for each group before aggregation
print("Counts for each group before aggregation:")
print(filtered_dice_scores_df.groupby(['gnn_layers', 'surf_hemi', 'surf_type']).size())

# Step 4: Group by gnn_layers, surf_hemi, and surf_type, and calculate the average dice score, standard deviation, and count for each group
grouped_dice_scores = filtered_dice_scores_df.groupby(['gnn_layers', 'surf_hemi', 'surf_type']).agg(
    average_dice_score=('test_dice_score', 'mean'),
    stdev_dice=('test_dice_score', 'std'),
    count=('test_dice_score', 'size')  # This will count the number of entries per group
).reset_index()

# Step 5: Replace values in the DataFrame
grouped_dice_scores['surf_hemi'] = grouped_dice_scores['surf_hemi'].replace({'lh': 'left hemisphere', 'rh': 'right hemisphere'})
grouped_dice_scores['surf_type'] = grouped_dice_scores['surf_type'].replace({'gm': 'pial'})

# Step 6: Save the grouped data to a CSV file
output_csv_path_grouped = '../csvs/grouped_dice_scores.csv'
grouped_dice_scores.to_csv(output_csv_path_grouped, index=False)

print("Grouped dice scores saved to", output_csv_path_grouped)
