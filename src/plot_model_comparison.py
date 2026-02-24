#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4: Model Performance Comparison Plots
Generates 3x3 comparison plots for CODMn, NH3-N, and TP across R², NMAE, and MAE metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for publication quality
plt.style.use('default')
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 13
rcParams['axes.linewidth'] = 1.0
rcParams['xtick.major.width'] = 1.0
rcParams['ytick.major.width'] = 1.0
rcParams['xtick.minor.width'] = 0.6
rcParams['ytick.minor.width'] = 0.6
rcParams['legend.frameon'] = True
rcParams['legend.fancybox'] = False
rcParams['axes.spines.top'] = True
rcParams['axes.spines.right'] = True
rcParams['axes.spines.bottom'] = True
rcParams['axes.spines.left'] = True

# Set publication style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
sns.set_palette("colorblind")

# Custom color palette for models
model_colors = {
    'RF': '#1f77b4',        # Blue
    'XGBoost': '#ff7f0e',   # Orange
    'DNN': '#2ca02c',       # Green
    'ISTA-LSTM': '#d62728'  # Red
}

def format_parameter_name(param_name):
    """Format parameter names with proper subscripts/superscripts"""
    if param_name == 'CODMn':
        return 'COD$_{\\mathrm{Mn}}$'
    elif param_name == 'NH3N':
        return 'NH$_3$-N'
    elif param_name == 'TP':
        return 'TP'
    else:
        return param_name

def create_boxplot_with_alpha(ax, data, positions, metric_name, alpha_value, param_idx):
    """Create boxplot with consistent alpha transparency"""
    parameter_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    median_color = '#d62728'
    whisker_color = '#1f77b4'
    box_alpha = alpha_value
    line_width = 0.6
    
    bp = ax.boxplot(data, positions=positions, vert=False, widths=0.04,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=parameter_colors[param_idx], alpha=box_alpha, linewidth=line_width),
                    whiskerprops=dict(color=whisker_color, linewidth=line_width, alpha=box_alpha),
                    capprops=dict(color=whisker_color, linewidth=line_width, alpha=box_alpha),
                    medianprops=dict(color=median_color, linewidth=line_width*1.5, alpha=1.0))
    
    return bp

def main():
    """Main function to generate Figure 4"""
    # Load data
    file_path = '../../data/processed/comparison.xlsx'
    xl = pd.ExcelFile(file_path)
    sheet_names = xl.sheet_names[:3]
    
    data_dict = {}
    for sheet in sheet_names:
        data_dict[sheet] = pd.read_excel(file_path, sheet_name=sheet)
    
    # Parameters
    metrics = ['R2', 'MAE', 'NMAE']
    metric_labels = ['R²', 'MAE (mg/L)', 'NMAE']
    methods = ['RF', 'XGBoost', 'DNN', 'ISTA-LSTM']
    feat_combs = sorted(data_dict[sheet_names[0]]['FeatComb'].unique())
    
    # Create 3x3 figure
    fig = plt.figure(figsize=(12, 18), dpi=300)
    gs = GridSpec(3, 3, figure=fig, width_ratios=[1, 1, 1], height_ratios=[1, 1, 1], 
                  hspace=0.06, wspace=0)
    
    # Create subplots
    all_axes = [
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])],
        [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[2, 2])]
    ]
    
    # Generate plots for each parameter and metric
    for param_idx, (param_name, param_data) in enumerate(zip(sheet_names, [data_dict[sheet] for sheet in sheet_names])):
        for metric_idx, metric in enumerate(metrics):
            ax = all_axes[param_idx][metric_idx]
            
            # Prepare data
            plot_data = []
            for method in methods:
                for feat_comb in feat_combs:
                    subset = param_data[(param_data['Method'] == method) & 
                                       (param_data['FeatComb'] == feat_comb)]
                    if len(subset) > 0:
                        plot_data.append(subset[metric].values)
            
            if len(plot_data) > 0:
                # Create positions
                positions = []
                current_y = 0
                model_spacing = 0.02
                scenario_spacing = 0.06
                
                for i, method in enumerate(methods):
                    for j, feat_comb in enumerate(feat_combs):
                        positions.append(current_y)
                        current_y += scenario_spacing
                    current_y += model_spacing
                
                # Plot boxplots
                create_boxplot_with_alpha(ax, plot_data, positions, metric, 0.7, param_idx)
            
            # Set labels
            ax.set_xlabel(metric_labels[metric_idx], fontsize=11, fontweight='bold')
            if metric_idx == 0:
                ax.set_ylabel(format_parameter_name(param_name), fontsize=12, fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    output_path = '../../results/figures/Fig4_ModelPerfComparison'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.savefig(f'{output_path}.pdf', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.savefig(f'{output_path}.svg', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    
    print(f"Figure 4 saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
