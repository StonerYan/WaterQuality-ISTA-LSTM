# High-Precision Water Quality Parameter Inversion using ISTA-LSTM

This repository contains the code and data for the paper: **"High-precision non-optically active water quality parameter inversion in small-to-medium rivers using ISTA-LSTM and spectral feature engineering"**.

## Overview

This project presents a comprehensive framework for monitoring non-optically active water quality parameters (CODMn, NH3-N, and TP) in small-to-medium rivers using exclusively Sentinel-2 satellite imagery. The Irregular Sequence Temporal Attention LSTM (ISTA-LSTM) architecture integrates spectral feature engineering to achieve high-precision water quality estimation.

### Key Features

- **Comprehensive Spectral Feature Engineering**: 6 feature categories including vegetation reference, temporal anomaly detection, and adjacency correction
- **ISTA-LSTM Architecture**: Handles sparse, irregularly-sampled satellite observations with attention mechanisms
- **High Accuracy**: CODMn (R² = 0.79), NH3-N (R² = 0.55), TP (R² = 0.69)
- **Interpretability**: SHAP-based analysis for understanding model predictions
- **Uncertainty Quantification**: Monte Carlo dropout for reliability assessment

## Project Structure

```
WaterQuality-ISTA-LSTM/
├── data/                           # Data directory
├── src/                            # Source code
│   ├── data_processing/            # Data preprocessing scripts
│   ├── models/                     # Model implementations
│   └── visualization/              # Plotting and visualization scripts
├── configs/                        # Configuration files
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/WaterQuality-ISTA-LSTM.git
cd WaterQuality-ISTA-LSTM
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

Process raw Sentinel-2 and water quality data:

```bash
python src/data_processing/process_sentinel2_data.py
python src/data_processing/extract_spectral_features.py
python src/data_processing/merge_water_quality_data.py
```

### 2. Model Training

Train ISTA-LSTM models for different water quality parameters:

```bash
# Train CODMn model with multiple random seeds
python src/models/train_codmn_multi_seed.py

# Train NH3-N model
python src/models/train_nh3n_multi_seed.py

# Train TP model
python src/models/train_tp_multi_seed.py
```

### 3. Generate Figures

Reproduce figures from the paper:

```bash
# Figure 4: Model performance comparison
python src/visualization/plot_model_comparison.py

# Figure 5: FC09 performance analysis
python src/visualization/plot_fc09_performance.py

# Figure 6-8: SHAP analysis
python src/visualization/plot_shap_category.py
python src/visualization/plot_shap_band.py
python src/visualization/plot_shap_detailed.py
```

## Data

The `data/` directory contains:

- **Raw data**: Water quality measurements and Sentinel-2 spectral data from 32 monitoring stations (2020-2025)
- **Processed data**: Feature-engineered datasets ready for model training

### Data Description

- `3-水质数据_processed.csv`: Processed water quality measurements (CODMn, NH3-N, TP)
- `52-spectral_water_quality.csv`: Merged spectral-water quality dataset
- `cgx_spectral_data.csv`: Independent validation data from Changguangxi Creek

## Model Architecture

The ISTA-LSTM model includes:

1. **Input Processing Layer**: Handles spectral features, temporal gaps, and seasonal information
2. **Temporal Attention Mechanism**: Computes attention weights for historical observations
3. **LSTM Component**: Multi-layer unidirectional LSTM for sequential modeling
4. **Feature Fusion**: Combines current and attention-weighted historical information
5. **Output Layer**: Parameter-specific prediction heads with uncertainty estimation

## Acknowledgments

- Sentinel-2 data provided by the European Space Agency (ESA)
- Water quality data from China National Surface Water Quality Automatic Monitoring Dataset
