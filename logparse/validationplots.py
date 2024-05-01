import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_epochs_validation(input_csv):
    # Load the data
    df = pd.read_csv(input_csv)

    # Prepare a column that combines model_type, solver, and gnn_layers for detailed legend information
    df['model_solver_layers'] = df.apply(lambda row: f"{row['model_type']}_{row['solver']}_{row['gnn_layers']}", axis=1)

    # Set the plot style
    sns.set(style="whitegrid")

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex='col', sharey='row')
    fig.suptitle('Validation Error Over Epochs by Model Configuration', fontsize=16)

    # List of unique hemispheres and surf_types
    hemispheres = df['hemisphere'].unique()
    surf_types = df['surf_type'].unique()

    # Plotting
    for i, hemi in enumerate(hemispheres):
        for j, stype in enumerate(surf_types):
            ax = axes[i, j]
            subset_df = df[(df['hemisphere'] == hemi) & (df['surf_type'] == stype)]

            # Plot each group in the subset
            if not subset_df.empty:
                sns.lineplot(data=subset_df, x='epochs', y='validation_error', hue='model_solver_layers', style='model_solver_layers',
                             markers=True, dashes=False, ax=ax, palette='viridis', legend='brief')
                ax.set_title(f'Hemisphere: {hemi}, Surf Type: {stype}')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Validation Error')

    # Adjust the layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Place a single legend outside of the subplots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1), title='Configurations')

    # Save the plot
    plt.savefig('validation_error_plots.png')
    plt.show()

# Usage
input_csv_path = 'results.csv'
plot_epochs_validation(input_csv_path)
