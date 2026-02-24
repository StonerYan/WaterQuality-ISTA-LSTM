# Model Documentation

This directory contains the implementation of the ISTA-LSTM (Irregular Sequence Temporal Attention LSTM) model and baseline models for water quality parameter inversion.

## Model Architecture

### ISTA-LSTM

The Irregular Sequence Temporal Attention LSTM is designed to handle sparse, irregularly-sampled satellite observations for water quality estimation.

#### Key Components:

1. **Input Processing Layer**
   - Spectral indices
   - Temporal gap information (time since last observation)
   - Seasonal features (sine/cosine transformations of day of year)

2. **Temporal Attention Mechanism**
   - Computes attention weights for historical observations
   - Query-key-value mechanism
   - Attention weights: α_i = softmax(Q · K_i^T / √d_k)

3. **LSTM Component**
   - Multi-layer unidirectional LSTM
   - Dropout regularization between layers
   - Sequential hidden state representations

4. **Feature Fusion**
   - Concatenation of current observation and attention-weighted history
   - h_fused = concat(h_current, h_attention)

5. **Output Layer**
   - Parameter-specific prediction heads
   - ReLU activation and dropout regularization
   - Separate heads for CODMn, NH3-N, and TP

#### Training Strategy:

- **Optimizer**: Adam with adaptive learning rate scheduling
- **Loss Function**: MSE + attention regularization + temporal smoothness
  - L = MSE(y_pred, y_true) + λ_att × H(α) + λ_temp × ||Δα||²
- **Early Stopping**: Based on validation performance
- **Multi-seed Training**: 10 random seeds (111-1110) for robust evaluation

### Baseline Models

1. **Random Forest (RF)**
   - Ensemble of decision trees
   - Hyperparameter tuning via Optuna
   - 5-fold cross-validation

2. **XGBoost**
   - Gradient boosting framework
   - Optimized with Optuna
   - Regularization parameters tuned

3. **Deep Neural Network (DNN)**
   - Multi-layer feedforward architecture
   - Fixed hyperparameters
   - Dropout for regularization

## File Structure

```
src/models/
├── ista_lstm_codmn.py          # ISTA-LSTM for CODMn
├── ista_lstm_nh3n.py           # ISTA-LSTM for NH3-N
├── ista_lstm_tp.py             # ISTA-LSTM for TP
├── train_codmn_multi_seed.py   # Multi-seed trainer for CODMn
├── train_nh3n_multi_seed.py    # Multi-seed trainer for NH3-N
├── train_tp_multi_seed.py      # Multi-seed trainer for TP
├── baseline_models.py          # RF, XGBoost, DNN implementations
└── utils.py                    # Utility functions
```

## Usage

### Single Model Training

```python
from ista_lstm_codmn import run_remote_sensing_test

# Train ISTA-LSTM for CODMn
results = run_remote_sensing_test(
    split_method='station',  # or 'random'
    enable_band_transforms=True,
    output_dir='results/codmn'
)
```

### Multi-Seed Training

```bash
# Train CODMn model with multiple seeds
python src/models/train_codmn_multi_seed.py

# Train NH3-N model
python src/models/train_nh3n_multi_seed.py

# Train TP model
python src/models/train_tp_multi_seed.py
```

## Model Performance

### CODMn (Permanganate Index)

| Model | R² | MAE (mg/L) | NMAE |
|-------|----|-----------| -----|
| RF | 0.54±0.03 | 0.63±0.03 | 0.209±0.008 |
| XGBoost | 0.60±0.04 | 0.58±0.03 | 0.191±0.009 |
| DNN | 0.68±0.03 | 0.51±0.02 | 0.169±0.006 |
| **ISTA-LSTM** | **0.79±0.02** | **0.40±0.02** | **0.135±0.006** |

### NH3-N (Ammonia Nitrogen)

| Model | R² | MAE (mg/L) | NMAE |
|-------|----|-----------| -----|
| RF | 0.32±0.02 | 0.129±0.004 | 0.562±0.019 |
| XGBoost | 0.34±0.04 | 0.124±0.004 | 0.542±0.020 |
| DNN | 0.37±0.03 | 0.119±0.005 | 0.521±0.024 |
| **ISTA-LSTM** | **0.55±0.04** | **0.097±0.005** | **0.431±0.015** |

### TP (Total Phosphorus)

| Model | R² | MAE (mg/L) | NMAE |
|-------|----|-----------| -----|
| RF | 0.38±0.02 | 0.023±0.001 | 0.220±0.009 |
| XGBoost | 0.43±0.04 | 0.022±0.001 | 0.209±0.010 |
| DNN | 0.56±0.04 | 0.019±0.001 | 0.184±0.008 |
| **ISTA-LSTM** | **0.69±0.03** | **0.016±0.001** | **0.148±0.003** |

## Hyperparameters

### ISTA-LSTM

```python
{
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.3,
    'attention_dim': 32,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 15
}
```

### Baseline Models

Hyperparameters are optimized using Optuna with 5-fold cross-validation. See `configs/model_config.yaml` for details.

## Model Interpretability

### SHAP Analysis

The models include SHAP (SHapley Additive exPlanations) analysis at three levels:

1. **Spectral Categories**: Identify which feature types contribute most
2. **Spectral Bands**: Reveal relevant spectral regions (B1-B12)
3. **Detailed Features**: Understand individual transformation contributions

### Uncertainty Quantification

Monte Carlo dropout provides uncertainty estimates:

```python
# Enable uncertainty quantification
model.eval()
predictions = []
for _ in range(100):
    with torch.no_grad():
        pred = model(x, enable_dropout=True)
        predictions.append(pred)

mean_pred = torch.mean(torch.stack(predictions), dim=0)
std_pred = torch.std(torch.stack(predictions), dim=0)
```

## References

1. LSTM: Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
2. Attention Mechanism: Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.
3. SHAP: Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS.
