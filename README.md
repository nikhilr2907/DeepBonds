# DeepBonds: LSTM Encoder-Decoder for Bond Yield Prediction

A deep learning approach for predicting US government bond yields using an LSTM encoder-decoder architecture with Professor Forcing training methodology.

## Overview

This project implements a sequence-to-sequence model for forecasting bond yields using historical bond data and macroeconomic indicators. The model employs:

- **LSTM Encoder-Decoder Architecture**: Captures temporal dependencies in financial time series
- **Principal Component Analysis (PCA)**: Dimensionality reduction for feature extraction
- **Teacher Forcing**: Training technique that uses ground truth as input during training
- **Professor Forcing**: Advanced adversarial training method that uses a discriminator to distinguish between teacher-forced and free-running behavior

## Features

- Multi-layer LSTM encoder-decoder with dropout regularization
- Integration of macroeconomic indicators (CPI, ISM Manufacturing Index)
- PCA-based feature engineering
- Teacher forcing and Professor Forcing training strategies
- Comprehensive visualization tools
- Modular and extensible codebase

## Project Structure

```
DeepBonds/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration and hyperparameters
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── models.py             # LSTM model architectures
│   ├── visualization.py      # Plotting and visualization utilities
│   └── utils.py              # Helper functions
├── main.py                   # Main training script
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (data paths)
├── .gitignore               # Git ignore file
└── README.md                # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/nikhilr2907/DeepBonds.git
cd DeepBonds
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with your data paths:
```env
BOND_DATA_PATH=/path/to/us-government-bond.csv
CPI_DATA_PATH=/path/to/CORESTICKM159SFRBATL.csv
ISM_DATA_PATH=/path/to/AMTMNO.csv
MODEL_SAVE_PATH=trained_model.pth
```

## Usage

### Basic Training

Train the model with default parameters:
```bash
python main.py
```

### Advanced Training Options

Train with custom hyperparameters:
```bash
python main.py --epochs 500 --learning-rate 0.006 --teacher-forcing-ratio 0.6
```

Train with Professor Forcing:
```bash
python main.py --use-professor-forcing --pf-epochs 500 --pf-learning-rate 0.003
```

Enable visualizations:
```bash
python main.py --visualize
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--epochs` | int | 500 | Number of training epochs |
| `--learning-rate` | float | 0.006 | Learning rate for optimizer |
| `--teacher-forcing-ratio` | float | 0.6 | Teacher forcing ratio (0.0 to 1.0) |
| `--dynamic-tf` | flag | False | Use dynamic teacher forcing (gradually decrease ratio) |
| `--use-professor-forcing` | flag | False | Enable Professor Forcing training |
| `--pf-epochs` | int | 500 | Number of epochs for Professor Forcing |
| `--pf-learning-rate` | float | 0.003 | Learning rate for Professor Forcing |
| `--visualize` | flag | False | Generate visualization plots |

## Model Architecture

### Encoder-Decoder Structure

1. **Encoder**: Multi-layer LSTM processes input sequences of macroeconomic features
2. **Decoder**: Two-stage decoder:
   - Primary decoder for initial prediction
   - Secondary decoder for autoregressive predictions
3. **Discriminator** (Professor Forcing only): Distinguishes between teacher-forced and free-running sequences

### Hyperparameters

- **Sequence Length**: 22 time steps
- **Input Size**: 3 (PCA components)
- **Hidden Size**: 50 LSTM units
- **Number of Layers**: 2
- **Batch Size**: 50
- **Dropout**: 0.5

## Training Methodology

### Teacher Forcing

Teacher forcing is a training technique where the model receives the ground truth output from the previous time step as input, rather than its own prediction. This helps stabilize training but can lead to exposure bias.

**Reference**: Williams, R. J., & Zipser, D. (1989). "A Learning Algorithm for Continually Running Fully Recurrent Neural Networks." *Neural Computation*, 1(2), 270-280.

### Professor Forcing

Professor Forcing extends teacher forcing by training the model adversarially. A discriminator network learns to distinguish between:
- **Teacher-forced behavior**: Model receives ground truth inputs
- **Free-running behavior**: Model uses its own predictions

This approach helps the model generalize better during inference when it must rely on its own predictions.

**Reference**: Lamb, A. M., Goyal, A., Zhang, Y., Zhang, S., Couville, A. C., & Bengio, Y. (2016). "Professor Forcing: A New Algorithm for Training Recurrent Networks." *Advances in Neural Information Processing Systems* (NeurIPS 2016), 4601-4609. [arXiv:1610.09038](https://arxiv.org/abs/1610.09038)

## Data

The model uses three primary data sources:

1. **US Government Bond Data**: Historical yield curves for various maturities
2. **Core CPI (CORESTICKM159SFRBATL)**: Inflation indicator from Federal Reserve
3. **ISM Manufacturing Index (AMTMNO)**: Economic activity indicator

Data should be in CSV format with a date column for temporal alignment.

## Results

The model achieves strong predictive performance on US 5-year bond yields:

- Trains with teacher forcing to stabilize learning
- Optional Professor Forcing for improved generalization
- Generates 22-step-ahead forecasts

## Evaluation Metrics

The model is evaluated using:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**

## Visualization

The project includes visualization tools for:
- Feature covariance matrix heatmaps
- PCA explained variance analysis
- Feature loadings for principal components
- Actual vs. predicted bond yield comparisons

Generated visualizations are saved as interactive HTML files using Plotly.


 
