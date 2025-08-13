#!/usr/bin/env python3
"""
Create a combined view of Pareto curves across all layers for easier comparison.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

# Set up the figure - larger to accommodate the bigger individual plots
fig = plt.figure(figsize=(24, 20))

# Layer names
layers = [
    'blocks_2_hook_resid_pre',
    'blocks_4_hook_resid_pre', 
    'blocks_6_hook_resid_pre'
]

layer_labels = ['Layer 2', 'Layer 4', 'Layer 6']

# Load and display each plot
for i, (layer, label) in enumerate(zip(layers, layer_labels)):
    img_path = f'plots/all_saes_pareto_{layer}.png'
    if Path(img_path).exists():
        img = mpimg.imread(img_path)
        ax = plt.subplot(3, 1, i+1)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'{label} Pareto Curves', fontsize=14, pad=10)

plt.suptitle('Pareto Curves Comparison: All SAE Types Across Layers\n(Filtered: L0 < 120, MSE < 0.006)', 
             fontsize=16, y=0.98)
plt.tight_layout()

# Save the combined figure
output_path = 'plots/all_saes_pareto_combined.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved combined view to: {output_path}")

plt.close()

# Also create a legend figure separately for clarity
fig_legend = plt.figure(figsize=(8, 2))
ax_legend = fig_legend.add_subplot(111)

# Define colors and markers as in main script
colors = {
    'ReLU': '#1f77b4',           # Blue
    'Gated': '#ff7f0e',          # Orange  
    'HC (input-dep)': '#2ca02c',    # Green
    'HC (input-indep)': '#d62728'   # Red
}

markers = {
    'ReLU': 'o',        # Circle
    'Gated': 's',       # Square
    'HC (input-dep)': '^',    # Triangle up
    'HC (input-indep)': 'D'   # Diamond
}

# Create legend entries
for label, color in colors.items():
    marker = markers[label]
    ax_legend.scatter([], [], color=color, marker=marker, s=100, label=label, alpha=0.7)
    ax_legend.plot([], [], color=color, linewidth=2, alpha=0.8, label=f'{label} Pareto')

ax_legend.legend(loc='center', ncol=4, frameon=True, fontsize=11,
                title='SAE Types and Pareto Frontiers', title_fontsize=12)
ax_legend.axis('off')

# Save legend
legend_path = 'plots/all_saes_pareto_legend.png'
plt.savefig(legend_path, dpi=150, bbox_inches='tight')
print(f"Saved legend to: {legend_path}")

print("\nColor and Shape Guide:")
print("=" * 40)
print("ReLU:              Blue circles (○)")
print("Gated:             Orange squares (■)")
print("HC (input-dep):    Green triangles (▲)")
print("HC (input-indep):  Red diamonds (♦)")
print("=" * 40)
print("\nPareto frontier points are highlighted with:")
print("- Larger size")
print("- Black border")
print("- Connected by lines of the same color") 