import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Read the data
file_path = 'cleaned_fastsurfer_measures.csv'
data = pd.read_csv(file_path)

# Separate the data into left and right hemispheres
lh_data = data[data['Hemisphere'] == 'LH']
rh_data = data[data['Hemisphere'] == 'RH']

# Separate the data into pial and white surfaces
pial_data = data[~data['Metric'].str.contains('White Surface')]
white_data = data[data['Metric'].str.contains('White Surface')]

# Calculate statistics
def calculate_statistics(df):
    grouped = df.groupby('Metric')['Score']
    means = grouped.mean().reset_index()
    stds = grouped.std().reset_index()
    stats = pd.merge(means, stds, on='Metric')
    stats.columns = ['Metric', 'Mean', 'StdDev']
    return stats

lh_pial_stats = calculate_statistics(lh_data)
rh_pial_stats = calculate_statistics(rh_data)

include_dice_for_white = False  # Change to True if dice scores are included for white surfaces

if include_dice_for_white:
    lh_white_stats = calculate_statistics(white_data)
    rh_white_stats = calculate_statistics(white_data)
else:
    lh_white_stats = calculate_statistics(white_data[~white_data['Metric'].str.contains('Dice')])
    rh_white_stats = calculate_statistics(white_data[~white_data['Metric'].str.contains('Dice')])

# Create the figure and GridSpec layout
fig = plt.figure(figsize=(20, 15))
gs = GridSpec(2, 2, figure=fig, wspace=0.05, hspace=0.05)

# Define a function to plot the tables
def plot_table(ax, df, title):
    ax.axis('off')
    if not df.empty:
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', edges='horizontal')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.auto_set_column_width(col=list(range(len(df.columns))))
        table.scale(1.2, 1.2)
        
        # Set thicker borders for outer cells
        for key, cell in table.get_celld().items():
            cell.set_linewidth(1.5 if key[0] == 0 or key[1] == -1 else 0.5)
            cell.set_edgecolor('black')
        
    ax.set_title(title, fontweight="bold", size=16)

# Plot each sub-table
ax1 = fig.add_subplot(gs[0, 0])
plot_table(ax1, lh_pial_stats, "Left Hemisphere Pial Surface")

ax2 = fig.add_subplot(gs[0, 1])
plot_table(ax2, rh_pial_stats, "Right Hemisphere Pial Surface")

ax3 = fig.add_subplot(gs[1, 0])
plot_table(ax3, lh_white_stats, "Left Hemisphere White Surface")

ax4 = fig.add_subplot(gs[1, 1])
plot_table(ax4, rh_white_stats, "Right Hemisphere White Surface")

# Adjust layout
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.tight_layout()

# Save the figure as an SVG file
output_svg_path = 'combined_table.svg'
plt.savefig(output_svg_path, format='svg')
print(f"Table saved as {output_svg_path}")

# Show the plot
plt.show()
