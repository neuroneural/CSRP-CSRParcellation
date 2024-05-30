import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_validation_dice_vs_epochs(input_csv, output_prefix):
    # Load the data
    df = pd.read_csv(input_csv)

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Plot validation dice vs epochs
    g = sns.FacetGrid(df, col="surf_hemi", hue="gnn_layers", sharey=False, height=6, aspect=1.5)
    g.map(sns.lineplot, "epoch", "validation_dice", marker="o")
    g.add_legend()
    g.set_axis_labels("Epochs", "Validation Dice")
    g.set_titles("{col_name}")
    g.fig.suptitle("Validation Dice vs Epochs", y=1.03)

    # Save plot as PNG and SVG
    plot_output_png = f"{output_prefix}_validation_dice_vs_epochs.png"
    plot_output_svg = f"{output_prefix}_validation_dice_vs_epochs.svg"
    g.savefig(plot_output_png)
    g.savefig(plot_output_svg)
    plt.close(g.fig)
    print(f"Plots saved as {plot_output_png} and {plot_output_svg}")

def save_top_dice_table(input_csv, output_prefix):
    # Load the data
    df = pd.read_csv(input_csv)

    # Find the row with the maximum validation dice for each group
    idx = df.groupby(['surf_hemi', 'gnn_layers'])['validation_dice'].idxmax()
    top_dice = df.loc[idx, ['surf_hemi', 'gnn_layers', 'epoch', 'validation_dice']]

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Create the table plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=top_dice.values, colLabels=top_dice.columns, loc='center', cellLoc='center', colColours=["#f2f2f2"]*top_dice.shape[1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Save table as PNG and SVG
    table_output_png = f"{output_prefix}_top_validation_dice.png"
    table_output_svg = f"{output_prefix}_top_validation_dice.svg"
    plt.savefig(table_output_png, format='png', bbox_inches='tight')
    plt.savefig(table_output_svg, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"Tables saved as {table_output_png} and {table_output_svg}")


# Usage
input_csv_path = '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/exp_vc_gnn_0/model/training_log_vertex_classification_vc.csv'
output_prefix = 'vc_results_table'
plot_validation_dice_vs_epochs(input_csv_path, output_prefix)
save_top_dice_table(input_csv_path, output_prefix)
