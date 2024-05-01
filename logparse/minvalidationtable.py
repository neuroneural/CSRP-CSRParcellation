import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_lowest_errors_with_epoch(input_csv, output_csv):
    # Read the data from CSV
    df = pd.read_csv(input_csv)

    # Ensure that all necessary fields are appropriately typed and handled.
    # For baseline models where 'gnn_layers' might be 'NA', ensure they're treated consistently.
    df['gnn_layers'].fillna('NA', inplace=True)  # Replace missing gnn_layers with 'NA'

    # Find the row with the minimum validation error for each group
    idx = df.groupby(['hemisphere', 'surf_type', 'model_type', 'solver', 'gnn_layers'])['validation_error'].idxmin()
    min_errors_with_epochs = df.loc[idx, ['hemisphere', 'surf_type', 'model_type', 'solver', 'gnn_layers', 'epochs', 'validation_error']]

    # Sort the result by validation error (ascending) to show the best results at the top
    min_errors_with_epochs.sort_values(by='validation_error', ascending=True, inplace=True)
    min_errors_with_epochs.to_csv(output_csv, index=False)
    print(f"Aggregated data with epochs written to {output_csv}")

# Usage
input_csv_path = 'results.csv'
output_csv_path = 'lowest_validation_errors_with_epochs.csv'
aggregate_lowest_errors_with_epoch(input_csv_path, output_csv_path)

def save_tables_as_svg(input_csv, output_svg_prefix):
    # Load the data
    df = pd.read_csv(input_csv)

    # Round the validation error column
    df['validation_error'] = df['validation_error'].round(7)

    # Set Seaborn style
    sns.set(style="whitegrid")

    # List of unique hemispheres and surf_types
    hemispheres = df['hemisphere'].unique()
    surf_types = df['surf_type'].unique()

    # Generate and save a table for each combination of hemisphere and surf_type
    for hemi in hemispheres:
        for stype in surf_types:
            subset_df = df[(df['hemisphere'] == hemi) & (df['surf_type'] == stype)]
            if not subset_df.empty:
                fig, ax = plt.subplots(figsize=(10, 4))  # Adjust the size as necessary
                ax.axis('tight')
                ax.axis('off')
                table = ax.table(cellText=subset_df.values, colLabels=subset_df.columns, loc='center', cellLoc='center', colColours=["#f2f2f2"]*subset_df.shape[1])
                table.auto_set_font_size(False)
                table.set_fontsize(10)  # Adjust font size
                table.scale(1.2, 1.2)  # Adjust scaling to match your needs

                output_svg = f"{output_svg_prefix}_{hemi}_{stype}.svg"
                plt.savefig(output_svg, format='svg', bbox_inches='tight')
                plt.close(fig)  # Close the figure to free memory
                print(f"Table saved as {output_svg}")

# Usage
input_csv_path = 'lowest_validation_errors_with_epochs.csv'
output_prefix = 'results_table'
save_tables_as_svg(input_csv_path, output_prefix)
