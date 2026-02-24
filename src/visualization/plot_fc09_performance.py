#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 5: FC09 Feature Combination Performance Analysis
Generates 3x3 performance plots for CODMn, NH3-N, and TP using FC09 combination
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

# Custom color palette
model_colors = {
    'RF': '#1f77b4',
    'XGBoost': '#ff7f0e',
    'DNN': '#2ca02c',
    'ISTA-LSTM': '#d62728'
}

def load_comparison_data(file_path):
    """Load data from comparison.xlsx file"""
    try:
        xl = pd.ExcelFile(file_path)
        sheet_names = xl.sheet_names[:3]
        
        data_dict = {}
        for sheet in sheet_names:
            data_dict[sheet] = pd.read_excel(file_path, sheet_name=sheet)
            print(f"Loaded {sheet} data: {len(data_dict[sheet])} rows")
        
        return data_dict, sheet_names
    except Exception as e:
        print(f"Failed to load data: {e}")
        return {}, []

def filter_fc09_data(data_dict):
    """Filter FC09 combination data"""
    fc09_data = {}
    
    for param, df in data_dict.items():
        fc09_subset = df[df['FeatComb'] == 'FC09'].copy()
        fc09_data[param] = fc09_subset
        print(f"{param} FC09 data: {len(fc09_subset)} rows")
    
    return fc09_data

def create_fc09_performance_plot(fc09_data, sheet_names, output_path):
    """Create FC09 performance comparison plot"""
    # Parameter labels
    param_labels = {
        sheet_names[0]: 'COD$_{\\mathrm{Mn}}$',
        sheet_names[1]: 'NH$_3$-N', 
        sheet_names[2]: 'TP'
    }
    
    metrics = ['R2', 'MAE', 'NMAE']
    metric_labels = ['R²', 'MAE (mg/L)', 'NMAE']
    methods = ['RF', 'XGBoost', 'DNN', 'ISTA-LSTM']
    
    # Create 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=300)
    fig.suptitle('FC09 Feature Combination Performance Analysis', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    subplot_labels = [
        ['(a)', '(b)', '(c)'],
        ['(d)', '(e)', '(f)'],
        ['(g)', '(h)', '(i)']
    ]
    
    for param_idx, param_name in enumerate(sheet_names):
        param_data = fc09_data[param_name]
        
        if len(param_data) == 0:
            print(f"Warning: No FC09 data for {param_name}")
            continue
            
        for metric_idx, metric in enumerate(metrics):
            ax = axes[param_idx, metric_idx]
            
            # Prepare data
            plot_data = []
            for method in methods:
                method_data = param_data[param_data['Method'] == method]
                if len(method_data) > 0:
                    plot_data.append(method_data[metric].values)
            
            if len(plot_data) > 0:
                # Create boxplot
                bp = ax.boxplot(plot_data, labels=methods, patch_artist=True)
                
                # Color boxes
                for patch, method in zip(bp['boxes'], methods):
                    patch.set_facecolor(model_colors[method])
                    patch.set_alpha(0.7)
                
                # Add scatter points
                for i, (method, data) in enumerate(zip(methods, plot_data), 1):
                    y = data
                    x = np.random.normal(i, 0.04, size=len(y))
                    ax.scatter(x, y, alpha=0.3, s=20, color=model_colors[method])
            
            # Set labels and title
            ax.set_xlabel('Model', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric_labels[metric_idx], fontsize=11, fontweight='bold')
            
            # Add subplot label
            ax.text(0.02, 0.98, subplot_labels[param_idx][metric_idx],
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   va='top', ha='left')
            
            # Set title for first row
            if param_idx == 0:
                ax.set_title(metric_labels[metric_idx], fontsize=12, fontweight='bold')
            
            # Set y-axis label for first column
            if metric_idx == 0:
                ax.set_ylabel(f"{param_labels[param_name]}\n{metric_labels[metric_idx]}", 
                            fontsize=11, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.savefig(f'{output_path}.pdf', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.savefig(f'{output_path}.svg', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    
    print(f"FC09 performance plot saved to {output_path}")
    plt.show()

def main():
    """Main function"""
    data_file = '../../data/processed/comparison.xlsx'
    output_path = '../../results/figures/Fig5_FC09_Performance_Analysis'
    
    print("Loading comparison.xlsx data...")
    
    # Load data
    data_dict, sheet_names = load_comparison_data(data_file)
    
    if not data_dict:
        print("Error: Unable to load data")
        return
    
    print(f"Successfully loaded {len(sheet_names)} parameters: {sheet_names}")
    
    # Filter FC09 data
    fc09_data = filter_fc09_data(data_dict)
    
    # Check if FC09 data exists
    total_fc09_records = sum(len(df) for df in fc09_data.values())
    if total_fc09_records == 0:
        print("Error: No FC09 data found")
        return
    
    print(f"Total {total_fc09_records} FC09 records found")
    
    # Create figure
    create_fc09_performance_plot(fc09_data, sheet_names, output_path)
    
    print("\nFigure description:")
    print("- 3x3 layout: 3 water quality parameters × 3 evaluation metrics")
    print("- Each subplot shows performance distribution of 4 models under FC09 combination")
    print("- Boxplots show performance distribution, scatter points show raw data")
    print("- Complies with scientific publication standards")

if __name__ == "__main__":
    main()
