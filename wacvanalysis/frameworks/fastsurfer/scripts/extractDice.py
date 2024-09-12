import pandas as pd

# Load the new CSV containing the dice scores
df = pd.read_csv('../csvs/fastsurfer_summary.csv')

# Filter the DataFrame to include only entries for the pial surfaces
pial_df = df[df['surf_type'] == 'pial']

# Assuming we want to keep only the relevant columns for appending
# Include 'std_Dice' column along with 'mean_Dice' and 'Hemisphere'
pial_df = pial_df[['Hemisphere', 'mean_Dice', 'std_Dice']]

# Rename columns to match your existing data
pial_df.rename(columns={
    'mean_Dice': 'average_dice_score',  # Correct renaming for average dice score
    'std_Dice': 'stdev_dice'  # Correct renaming for standard deviation of dice score
}, inplace=True)

# Add additional columns if necessary, like 'Vertex Mapping', 'Training Set', etc., with default values or mapped based on some logic
pial_df['Vertex Mapping'] = 'asymmetric'  # Example default value
pial_df['Training Set'] = 'cortexode'  # Example default value
pial_df['Testing Ground Truth'] = 'freesurfer'  # Example default value
pial_df['framework'] = 'example_framework_csrvc'  # Placeholder, replace with actual logic if necessary

# Save the filtered and formatted data to a new CSV, or append directly if that's the workflow
pial_df.to_csv('../csvs/updated_dice_scores.csv', index=False)

print("Filtered and prepared dice scores saved for pial surfaces.")
