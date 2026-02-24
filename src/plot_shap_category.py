#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP Spectral Category Importance and Partial Dependence Analysis
Generates 3x3 plots showing spectral category importance and partial dependence for water quality parameters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from pathlib import Path

# Configure matplotlib for publication quality
plt.style.use('default')
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 12
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

def load_importance_data():
    """Load spectral category importance data for three water quality parameters"""
    importance_files = {
        'NH$_3$-N': '../../results/shap_analysis/nh3n/aggregated_importance_spectral_type.csv',
        'TP': '../../results/shap_analysis/tp/aggregated_importance_spectral_type.csv',
        'COD$_{\\mathrm{Mn}}$': '../../results/shap_analysis/codmn/aggregated_importance_spectral_type.csv'
    }
    
    importance_data = {}
    for param, file_path in importance_files.items():
        if os.path.exists(file_path):
            importance_data[param] = pd.read_csv(file_path)
    
    return importance_data

def load_partial_dependence_data():
    """Load partial dependence data"""
    pd_files = {
        'NH$_3$-N': '../../results/shap_analysis/nh3n/partial_dependence_scatter_data_spectral_type.csv',
        'TP': '../../results/shap_analysis/tp/partial_dependence_scatter_data_spectral_type.csv',
        'COD$_{\\mathrm{Mn}}$': '../../results/shap_analysis/codmn/partial_dependence_scatter_data_spectral_type.csv'
    }
    
    pd_data = {}
    for param, file_path in pd_files.items():
        if os.path.exists(file_path):
            pd_data[param] = pd.read_csv(file_path)
    
    return pd_data

def get_top_features(importance_data, n_features=2):
    """Get top n features for each parameter"""
    top_features = {}
    for param, df in importance_data.items():
        df_sorted = df.sort_values('mean_importance', ascending=False)
        top_features[param] = df_sorted.head(n_features)['feature'].tolist()
    
    return top_features

def create_binned_partial_dependence(feature_data, n_bins=20):
    """Create binned partial dependence data"""
    # Remove outliers using IQR method
    Q1 = feature_data['feature_value_original'].quantile(0.25)
    Q3 = feature_data['feature_value_original'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = feature_data[
        (feature_data['feature_value_original'] >= lower_bound) & 
        (feature_data['feature_value_original'] <= upper_bound)
    ].copy()
    
    # Create bins
    filtered_data['bin'] = pd.cut(filtered_data['feature_value_original'], 
                                 bins=n_bins, include_lowest=True)
    
    # Calculate statistics for each bin
    bin_stats = filtered_data.groupby('bin', observed=False).agg({
        'feature_value_original': ['mean', 'count'],
        'shap_value': ['mean', 'std']
    }).reset_index()
    
    bin_stats.columns = ['bin', 'feature_mean', 'count', 'shap_mean', 'shap_std']
    bin_stats['shap_se'] = bin_stats['shap_std'] / np.sqrt(bin_stats['count'])
    
    # Keep bins with sufficient samples (at least 5)
    bin_stats = bin_stats[bin_stats['count'] >= 5]
    
    return bin_stats

def plot_importance_bar(ax, importance_df, param_name, colors):
    """Plot importance bar chart"""
    top_features = importance_df.head(5)
    
    # Feature name mapping
    feature_labels = {
        'veg_spectral': 'Veg\\nSpectral',
        'anomaly_spectral': 'Ano\\nSpectral', 
        'station_spectral': 'Sta\\nSpectral',
        'veg_corrected_spectral': 'Veg-Adj\\nSpectral',
        'time_features': 'Time'
    }
    
    labels = [feature_labels.get(f, f.replace('_', '\\n').title()) for f in top_features['feature']]
    
    bars = ax.bar(range(len(top_features)), top_features['mean_importance'], 
                  yerr=top_features['std_importance'], 
                  color=colors[param_name], alpha=0.7, 
                  capsize=4, ecolor='black')
    
    ax.set_xticks(range(len(top_features)))
    ax.set_xticklabels(labels, fontsize=10, ha='center')
    ax.set_ylabel('SHAP Importance', fontweight='bold')
    ax.set_title(f'{param_name}', fontweight='bold', fontsize=13)
    
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    y_max = (top_features['mean_importance'] + top_features['std_importance']).max()
    ax.set_ylim(0, y_max * 1.1)

def plot_partial_dependence(ax, pd_data, feature_name, param_name, colors):
    """Plot partial dependence"""
    feature_data = pd_data[pd_data['feature_name'] == feature_name].copy()
    
    if len(feature_data) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return
    
    bin_stats = create_binned_partial_dependence(feature_data, n_bins=20)
    
    if len(bin_stats) > 0:
        # Plot scatter with transparency
        ax.scatter(feature_data['feature_value_original'], 
                  feature_data['shap_value'],
                  alpha=0.1, s=10, color='gray', label='Individual samples')
        
        # Plot binned averages with error bars
        ax.errorbar(bin_stats['feature_mean'], bin_stats['shap_mean'],
                   yerr=bin_stats['shap_se'], 
                   fmt='o-', color=colors[param_name], linewidth=2,
                   markersize=6, capsize=3, capthick=1.5,
                   label='Binned average ± SE', alpha=0.9)
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel(f'{feature_name}', fontweight='bold')
        ax.set_ylabel('SHAP Value', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

def create_3x3_plot():
    """Create 3x3 plot layout"""
    # Load data
    importance_data = load_importance_data()
    pd_data = load_partial_dependence_data()
    top_features = get_top_features(importance_data, n_features=2)
    
    # Color scheme for parameters
    param_colors = {
        'NH$_3$-N': '#1f77b4',
        'TP': '#ff7f0e',
        'COD$_{\\mathrm{Mn}}$': '#2ca02c'
    }
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=300)
    fig.suptitle('Spectral Category SHAP Importance and Partial Dependence Analysis',
                 fontsize=14, fontweight='bold', y=0.98)
    
    param_order = ['COD$_{\\mathrm{Mn}}$', 'NH$_3$-N', 'TP']
    
    for row_idx, param in enumerate(param_order):
        if param not in importance_data:
            continue
        
        # Plot importance (column 0)
        plot_importance_bar(axes[row_idx, 0], importance_data[param], param, param_colors)
        
        # Plot partial dependence for top 2 features (columns 1-2)
        if param in top_features:
            for col_idx, feature in enumerate(top_features[param][:2], 1):
                if param in pd_data:
                    plot_partial_dependence(axes[row_idx, col_idx], pd_data[param], 
                                          feature, param, param_colors)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_path = '../../results/figures/Fig6_SHAP_Category_Analysis'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.savefig(f'{output_path}.pdf', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    
    print(f"Figure 6 saved to {output_path}")
    plt.show()

def main():
    """Main function"""
    print("Generating Figure 6: SHAP Category Analysis...")
    create_3x3_plot()
    print("Figure generation completed!")

if __name__ == "__main__":
    main()
