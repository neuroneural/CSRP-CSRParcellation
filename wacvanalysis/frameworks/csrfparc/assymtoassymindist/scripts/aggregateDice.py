import pandas as pd

# Step 1: Read the CSV file containing standard deviations and identify subjects to exclude
stdev_df = pd.read_csv('/data/users2/washbee/CortexODE-CSRFusionNet/wacvanalysis/frameworks/freesurfer/mwremoved/lowerstdevs.csv')
excluded_subjects = stdev_df[stdev_df['min_stdevfrommean'] < 2]['subject_id'].tolist()

# Step 2: Read the CSV file containing dice scores
dice_scores_df = pd.read_csv('../csvs/test_results_vertex_classification_vc.csv')

# Step 3: Exclude the subjects identified in step 1 from the dice scores
filtered_dice_scores_df = dice_scores_df[~dice_scores_df['subject_id'].isin(excluded_subjects)]

# Step 4: Group by gnn_layers, surf_hemi, and surf_type, and calculate the average dice score for each group
grouped_dice_scores = filtered_dice_scores_df.groupby(['gnn_layers', 'surf_hemi', 'surf_type'])['test_dice_score'].mean().reset_index()
grouped_dice_scores.rename(columns={'test_dice_score': 'average_dice_score'}, inplace=True)

# Step 5: Output the grouped and averaged results to a new CSV file
grouped_dice_scores.to_csv('../csvs/aggregated_dice.csv', index=False)

print("Output CSV with grouped and averaged dice scores has been saved.")
