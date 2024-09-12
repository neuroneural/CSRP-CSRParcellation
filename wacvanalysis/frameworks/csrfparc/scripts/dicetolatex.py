import pandas as pd

# Load the DataFrame from a CSV file
df = pd.read_csv('../csvs/organized_dice_scores_v2.csv')  # Ensure the path to your CSV file is correct

# Ensure you only select and order columns that exist and are needed
final_df = df[['framework', 'gnn_layers', 'average_dice_score', 'stdev_dice', 'Vertex Mapping','Training Input Surface','Testing Input Surface','surf_hemi']]

# Format the numerical columns to three significant digits
final_df['average_dice_score'] = final_df['average_dice_score'].apply(lambda x: f"{x:.3f}")
final_df['stdev_dice'] = final_df['stdev_dice'].apply(lambda x: f"{x:.3f}")

# Convert the DataFrame to LaTeX code
latex_table = final_df.to_latex(index=False, header=[
    'Framework', 'GNN Layers', 'Average Dice Score', 'Standard Deviation','Vertex Mapping','Training Input Surface','Testing Input Surface','Hemisphere'
], column_format='lcccc', escape=False)

# Print or save the LaTeX table
print(latex_table)
