#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentinel-2 Data Processing
Extracts spectral features from Sentinel-2 imagery for water quality monitoring
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def extract_spectral_bands(data_path):
    """
    Extract and process Sentinel-2 spectral bands
    
    Args:
        data_path: Path to raw Sentinel-2 data
        
    Returns:
        DataFrame with processed spectral bands
    """
    print(f"Processing Sentinel-2 data from {data_path}")
    
    # Load raw data
    df = pd.read_csv(data_path)
    
    # Sentinel-2 bands (B1-B12)
    band_columns = [f'B{i}' for i in range(1, 13)]
    
    # Extract station spectral bands
    station_bands = df[band_columns].copy()
    station_bands.columns = [f'station_{col}' for col in band_columns]
    
    return station_bands

def extract_vegetation_reference(data_path, buffer_radius=1000):
    """
    Extract reference vegetation spectral characteristics
    
    Args:
        data_path: Path to vegetation spectral data
        buffer_radius: Buffer radius in meters (default: 1000m)
        
    Returns:
        DataFrame with vegetation reference bands
    """
    print(f"Extracting vegetation reference within {buffer_radius}m buffer")
    
    # Load vegetation data
    df = pd.read_csv(data_path)
    
    # Filter vegetation pixels (NDVI > 0.3)
    if 'NDVI' in df.columns:
        veg_df = df[df['NDVI'] > 0.3].copy()
    else:
        # Calculate NDVI if not present
        nir = df['B8']
        red = df['B4']
        ndvi = (nir - red) / (nir + red + 1e-8)
        veg_df = df[ndvi > 0.3].copy()
    
    # Calculate mean vegetation spectral values
    band_columns = [f'B{i}' for i in range(1, 13)]
    veg_means = veg_df[band_columns].mean()
    veg_means.index = [f'veg_{col}' for col in band_columns]
    
    return veg_means

def calculate_temporal_anomaly(data_path):
    """
    Calculate temporal spectral anomalies
    
    Args:
        data_path: Path to time series spectral data
        
    Returns:
        DataFrame with temporal anomaly bands
    """
    print("Calculating temporal spectral anomalies")
    
    df = pd.read_csv(data_path)
    
    # Calculate long-term median for each band
    band_columns = [f'B{i}' for i in range(1, 13)]
    
    anomaly_bands = pd.DataFrame()
    for band in band_columns:
        if band in df.columns:
            median_val = df[band].median()
            anomaly_bands[f'anomaly_{band}'] = df[band] - median_val
    
    return anomaly_bands

def calculate_vegetation_adjusted(station_bands, veg_reference):
    """
    Calculate vegetation-adjusted spectral bands
    
    Args:
        station_bands: DataFrame with station spectral bands
        veg_reference: Series with vegetation reference values
        
    Returns:
        DataFrame with vegetation-adjusted bands
    """
    print("Calculating vegetation-adjusted spectral bands")
    
    veg_adjusted = pd.DataFrame()
    
    for i in range(1, 13):
        station_col = f'station_B{i}'
        veg_col = f'veg_B{i}'
        
        if station_col in station_bands.columns and veg_col in veg_reference.index:
            veg_adjusted[f'veg_corrected_B{i}'] = (
                station_bands[station_col] - veg_reference[veg_col]
            )
    
    return veg_adjusted

def apply_spectral_transformations(bands_df, transform_types=None):
    """
    Apply mathematical transformations to spectral bands
    
    Args:
        bands_df: DataFrame with spectral bands
        transform_types: List of transformation types to apply
        
    Returns:
        DataFrame with transformed features
    """
    if transform_types is None:
        transform_types = ['log', 'sqrt', 'square']
    
    print(f"Applying spectral transformations: {transform_types}")
    
    transformed_df = bands_df.copy()
    
    for col in bands_df.columns:
        if 'log' in transform_types:
            # Logarithmic transformation (add small value to avoid log(0))
            transformed_df[f'{col}_log'] = np.log(bands_df[col] + 1e-8)
        
        if 'sqrt' in transform_types:
            # Square root transformation
            transformed_df[f'{col}_sqrt'] = np.sqrt(np.abs(bands_df[col]))
        
        if 'square' in transform_types:
            # Square transformation
            transformed_df[f'{col}_square'] = bands_df[col] ** 2
    
    return transformed_df

def process_sentinel2_data(raw_data_path, output_path, options=None):
    """
    Main processing pipeline for Sentinel-2 data
    
    Args:
        raw_data_path: Path to raw Sentinel-2 data
        output_path: Path to save processed data
        options: Dictionary of processing options
    """
    if options is None:
        options = {
            'extract_vegetation': True,
            'calculate_anomaly': True,
            'vegetation_adjustment': True,
            'apply_transforms': True
        }
    
    print("=" * 60)
    print("Sentinel-2 Data Processing Pipeline")
    print("=" * 60)
    
    # Step 1: Extract station spectral bands
    station_bands = extract_spectral_bands(raw_data_path)
    print(f"✓ Extracted {len(station_bands.columns)} station spectral bands")
    
    # Step 2: Extract vegetation reference (if enabled)
    if options.get('extract_vegetation', True):
        veg_reference = extract_vegetation_reference(raw_data_path)
        print(f"✓ Extracted vegetation reference ({len(veg_reference)} bands)")
    else:
        veg_reference = None
    
    # Step 3: Calculate temporal anomalies (if enabled)
    if options.get('calculate_anomaly', True):
        anomaly_bands = calculate_temporal_anomaly(raw_data_path)
        print(f"✓ Calculated temporal anomalies ({len(anomaly_bands.columns)} bands)")
    else:
        anomaly_bands = None
    
    # Step 4: Calculate vegetation-adjusted bands (if enabled)
    if options.get('vegetation_adjustment', True) and veg_reference is not None:
        veg_adjusted = calculate_vegetation_adjusted(station_bands, veg_reference)
        print(f"✓ Calculated vegetation-adjusted bands ({len(veg_adjusted.columns)} bands)")
    else:
        veg_adjusted = None
    
    # Step 5: Apply spectral transformations (if enabled)
    if options.get('apply_transforms', True):
        all_bands = pd.concat([
            station_bands,
            veg_adjusted if veg_adjusted is not None else pd.DataFrame(),
            anomaly_bands if anomaly_bands is not None else pd.DataFrame()
        ], axis=1)
        
        transformed_bands = apply_spectral_transformations(all_bands)
        print(f"✓ Applied spectral transformations ({len(transformed_bands.columns)} features)")
    else:
        transformed_bands = pd.concat([
            station_bands,
            veg_adjusted if veg_adjusted is not None else pd.DataFrame(),
            anomaly_bands if anomaly_bands is not None else pd.DataFrame()
        ], axis=1)
    
    # Save processed data
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    transformed_bands.to_csv(output_file, index=False)
    
    print("=" * 60)
    print(f"✓ Processing complete! Saved to {output_file}")
    print(f"  Total features: {len(transformed_bands.columns)}")
    print(f"  Total samples: {len(transformed_bands)}")
    print("=" * 60)

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Process Sentinel-2 data for water quality monitoring'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to raw Sentinel-2 data (CSV format)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save processed data'
    )
    parser.add_argument(
        '--no-vegetation',
        action='store_true',
        help='Disable vegetation reference extraction'
    )
    parser.add_argument(
        '--no-anomaly',
        action='store_true',
        help='Disable temporal anomaly calculation'
    )
    parser.add_argument(
        '--no-adjustment',
        action='store_true',
        help='Disable vegetation adjustment'
    )
    parser.add_argument(
        '--no-transforms',
        action='store_true',
        help='Disable spectral transformations'
    )
    
    args = parser.parse_args()
    
    options = {
        'extract_vegetation': not args.no_vegetation,
        'calculate_anomaly': not args.no_anomaly,
        'vegetation_adjustment': not args.no_adjustment,
        'apply_transforms': not args.no_transforms
    }
    
    process_sentinel2_data(args.input, args.output, options)

if __name__ == '__main__':
    main()
