#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISTA-LSTM Model Training for NH3-N Parameter
Multi-seed experiment runner for robust performance evaluation
"""

import os
import sys
import pandas as pd
from pathlib import Path
import shutil

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import main training function
from ista_lstm_nh3n import run_remote_sensing_test

def run_multi_seed_experiment():
    """
    Run multi-seed experiments for NH3-N parameter estimation
    Random seeds: 111, 222, 333, ..., 999, 1110
    """
    # Define random seed list
    seeds = list(range(111, 1111, 111)) 
    
    # Create main output directory
    main_output_dir = Path('../../results/nh3n_multi_seed')
    main_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Storage for combined CSV data
    combined_csv_data = {
        'comparison': [],
        'uncertainty_metrics': [],
        'uncertainty_scatter_single': [],
        'uncertainty_scatter_sequence': []
    }
    
    print(f"Starting multi-seed experiment, seed list: {seeds}")
    print(f"Main output directory: {main_output_dir.absolute()}")
    print("=" * 80)
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n[{i}/{len(seeds)}] Running seed: {seed}")
        print("-" * 50)
        
        # Create output directory for current seed
        seed_output_dir = main_output_dir / f'seed_{seed}'
        seed_output_dir.mkdir(exist_ok=True)
        
        # Modify global random seed
        import ista_lstm_nh3n
        ista_lstm_nh3n.RANDOM_SEED = seed
        ista_lstm_nh3n.OUTPUT_DIR = f'ista_lstm_results_nh3n_{seed}'
        
        try:
            # Run experiment
            results = run_remote_sensing_test(
                split_method='random',
                enable_band_transforms=True,
                output_dir=str(seed_output_dir)
            )
            
            print(f"Seed {seed} experiment completed")
            
            # Process CSV files - add seed column and collect data
            comparison_csv = seed_output_dir / 'model_comparison.csv'
            if comparison_csv.exists():
                df = pd.read_csv(comparison_csv)
                df['seed'] = seed
                df.to_csv(comparison_csv, index=False)
                combined_csv_data['comparison'].append(df)
            
            uncertainty_csv = seed_output_dir / 'uncertainty_metrics.csv'
            if uncertainty_csv.exists():
                df = pd.read_csv(uncertainty_csv)
                df['seed'] = seed
                df.to_csv(uncertainty_csv, index=False)
                combined_csv_data['uncertainty_metrics'].append(df)
            
            # Collect uncertainty scatter data (separate single and sequence modes)
            for mode in ['single', 'sequence']:
                scatter_csv = seed_output_dir / f'uncertainty_scatter_{mode}.csv'
                if scatter_csv.exists():
                    df = pd.read_csv(scatter_csv)
                    df['seed'] = seed
                    df['mode'] = mode
                    df.to_csv(scatter_csv, index=False)
                    combined_csv_data[f'uncertainty_scatter_{mode}'].append(df)
            
            # Rename image files with seed identifier
            for file_path in seed_output_dir.glob('*.png'):
                if f'_seed_{seed}' not in file_path.stem:
                    new_name = f"{file_path.stem}_seed_{seed}{file_path.suffix}"
                    new_path = file_path.parent / new_name
                    file_path.rename(new_path)
                    
        except Exception as e:
            print(f"Seed {seed} failed: {str(e)}")
            continue
    
    # Merge all CSV files
    print("\n" + "=" * 50)
    print("Merging CSV files...")
    print("=" * 50)
    
    if combined_csv_data['comparison']:
        combined_comparison = pd.concat(combined_csv_data['comparison'], ignore_index=True)
        combined_comparison.to_csv(
            main_output_dir / 'combined_comparison_results.csv',
            index=False
        )
        print(f"Saved: {main_output_dir / 'combined_comparison_results.csv'}")
    
    if combined_csv_data['uncertainty_metrics']:
        combined_uncertainty = pd.concat(combined_csv_data['uncertainty_metrics'], ignore_index=True)
        combined_uncertainty.to_csv(
            main_output_dir / 'combined_uncertainty_metrics.csv',
            index=False
        )
        print(f"Saved: {main_output_dir / 'combined_uncertainty_metrics.csv'}")
    
    # Merge uncertainty scatter data
    all_scatter_data = []
    for mode in ['single', 'sequence']:
        if combined_csv_data[f'uncertainty_scatter_{mode}']:
            mode_data = pd.concat(combined_csv_data[f'uncertainty_scatter_{mode}'], ignore_index=True)
            all_scatter_data.append(mode_data)
    
    if all_scatter_data:
        combined_scatter = pd.concat(all_scatter_data, ignore_index=True)
        combined_scatter.to_csv(
            main_output_dir / 'combined_uncertainty_scatter_data.csv',
            index=False
        )
        print(f"Saved: {main_output_dir / 'combined_uncertainty_scatter_data.csv'}")
    
    print(f"\n{'='*80}")
    print(f"Multi-seed experiment completed!")
    print(f"Main output directory: {main_output_dir.absolute()}")
    print(f"Contains results from {len(seeds)} seeds")
    print(f"Individual seed results saved in subdirectories")
    print(f"Combined CSV files saved in main directory")
    print(f"Image files distinguished by seed in filename")
    print(f"{'='*80}")

if __name__ == "__main__":
    run_multi_seed_experiment()
