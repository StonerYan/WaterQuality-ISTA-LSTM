#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Water Quality Data Merging
Merges water quality measurements with Sentinel-2 spectral data based on temporal matching
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import argparse

def load_water_quality_data(wq_path):
    """
    Load and preprocess water quality data
    
    Args:
        wq_path: Path to water quality data file
        
    Returns:
        DataFrame with water quality measurements
    """
    print(f"Loading water quality data from {wq_path}")
    
    df = pd.read_csv(wq_path)
    
    # Convert time column to datetime
    time_col = '监测时间' if '监测时间' in df.columns else 'time'
    df[time_col] = pd.to_datetime(df[time_col])
    
    print(f"✓ Loaded {len(df)} water quality records")
    return df

def load_spectral_data(spectral_path):
    """
    Load processed spectral data
    
    Args:
        spectral_path: Path to processed spectral data
        
    Returns:
        DataFrame with spectral features
    """
    print(f"Loading spectral data from {spectral_path}")
    
    df = pd.read_csv(spectral_path)
    
    # Convert time column to datetime if exists
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    elif 'date' in df.columns:
        df['time'] = pd.to_datetime(df['date'])
    
    print(f"✓ Loaded {len(df)} spectral records")
    return df

def temporal_matching(wq_df, spectral_df, window_hours=12):
    """
    Match water quality measurements with spectral data using temporal window
    
    Args:
        wq_df: Water quality DataFrame
        spectral_df: Spectral data DataFrame
        window_hours: Temporal matching window (±hours)
        
    Returns:
        Merged DataFrame
    """
    print(f"Performing temporal matching (±{window_hours} hours window)")
    
    matched_records = []
    
    # Get time columns
    wq_time_col = '监测时间' if '监测时间' in wq_df.columns else 'time'
    spectral_time_col = 'time'
    
    for idx, spectral_row in spectral_df.iterrows():
        spectral_time = spectral_row[spectral_time_col]
        
        # Define time window
        time_min = spectral_time - timedelta(hours=window_hours)
        time_max = spectral_time + timedelta(hours=window_hours)
        
        # Find water quality measurements within window
        wq_subset = wq_df[
            (wq_df[wq_time_col] >= time_min) & 
            (wq_df[wq_time_col] <= time_max)
        ]
        
        if len(wq_subset) > 0:
            # Average water quality measurements within window
            wq_avg = wq_subset.select_dtypes(include=[np.number]).mean()
            
            # Combine spectral and water quality data
            matched_row = pd.concat([
                spectral_row,
                wq_avg
            ])
            matched_row['spectral_time'] = spectral_time
            matched_row['wq_time'] = wq_subset[wq_time_col].mean()
            matched_row['n_wq_samples'] = len(wq_subset)
            
            matched_records.append(matched_row)
    
    matched_df = pd.DataFrame(matched_records)
    
    print(f"✓ Matched {len(matched_df)} records")
    return matched_df

def quality_control(df, parameters=['CODMn', 'NH3N', 'TP'], mad_threshold=3.5):
    """
    Apply quality control to remove outliers
    
    Args:
        df: DataFrame with merged data
        parameters: List of water quality parameters to check
        mad_threshold: MAD Z-score threshold for outlier detection
        
    Returns:
        DataFrame with outliers removed
    """
    print(f"Applying quality control (MAD threshold: {mad_threshold})")
    
    # Map Chinese parameter names to English if needed
    param_map = {
        '高锰酸盐指数': 'CODMn',
        '氨氮': 'NH3N',
        '总磷': 'TP'
    }
    
    # Rename columns if needed
    for cn_name, en_name in param_map.items():
        if cn_name in df.columns and en_name not in df.columns:
            df = df.rename(columns={cn_name: en_name})
    
    initial_count = len(df)
    
    for param in parameters:
        if param not in df.columns:
            continue
        
        # Calculate MAD Z-score
        median = df[param].median()
        mad = np.median(np.abs(df[param] - median))
        mad_zscore = np.abs((df[param] - median) / (mad + 1e-8))
        
        # Remove outliers
        outlier_mask = mad_zscore > mad_threshold
        n_outliers = outlier_mask.sum()
        
        df = df[~outlier_mask]
        
        print(f"  {param}: removed {n_outliers} outliers")
    
    final_count = len(df)
    print(f"✓ Quality control complete: {initial_count} → {final_count} records")
    
    return df

def merge_water_quality_spectral(wq_path, spectral_path, output_path, options=None):
    """
    Main pipeline for merging water quality and spectral data
    
    Args:
        wq_path: Path to water quality data
        spectral_path: Path to spectral data
        output_path: Path to save merged data
        options: Dictionary of processing options
    """
    if options is None:
        options = {
            'window_hours': 12,
            'apply_qc': True,
            'mad_threshold': 3.5
        }
    
    print("=" * 60)
    print("Water Quality - Spectral Data Merging Pipeline")
    print("=" * 60)
    
    # Load data
    wq_df = load_water_quality_data(wq_path)
    spectral_df = load_spectral_data(spectral_path)
    
    # Temporal matching
    merged_df = temporal_matching(
        wq_df, 
        spectral_df, 
        window_hours=options.get('window_hours', 12)
    )
    
    # Quality control
    if options.get('apply_qc', True):
        merged_df = quality_control(
            merged_df,
            mad_threshold=options.get('mad_threshold', 3.5)
        )
    
    # Save merged data
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_file, index=False)
    
    print("=" * 60)
    print(f"✓ Merging complete! Saved to {output_file}")
    print(f"  Total records: {len(merged_df)}")
    print(f"  Features: {len(merged_df.columns)}")
    print("=" * 60)

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Merge water quality and spectral data'
    )
    parser.add_argument(
        '--wq-data',
        type=str,
        required=True,
        help='Path to water quality data (CSV format)'
    )
    parser.add_argument(
        '--spectral-data',
        type=str,
        required=True,
        help='Path to spectral data (CSV format)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save merged data'
    )
    parser.add_argument(
        '--window-hours',
        type=int,
        default=12,
        help='Temporal matching window in hours (default: 12)'
    )
    parser.add_argument(
        '--no-qc',
        action='store_true',
        help='Disable quality control'
    )
    parser.add_argument(
        '--mad-threshold',
        type=float,
        default=3.5,
        help='MAD Z-score threshold for outlier detection (default: 3.5)'
    )
    
    args = parser.parse_args()
    
    options = {
        'window_hours': args.window_hours,
        'apply_qc': not args.no_qc,
        'mad_threshold': args.mad_threshold
    }
    
    merge_water_quality_spectral(
        args.wq_data,
        args.spectral_data,
        args.output,
        options
    )

if __name__ == '__main__':
    main()
