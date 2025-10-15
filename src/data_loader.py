"""Data loading and preprocessing for bond yield prediction"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch


def load_data():
    """Load and perform initial processing of bond and economic data."""

    # Load main bond data
    bond_data_path = os.getenv('BOND_DATA_PATH', '/content/sample_data/us-government-bond.csv')
    df = pd.read_csv(bond_data_path)
    df.dropna(inplace=True)

    # Load CPI data
    cpi_data_path = os.getenv('CPI_DATA_PATH', '/content/sample_data/CORESTICKM159SFRBATL.csv')
    cpi = pd.read_csv(cpi_data_path)
    cpi["DATE"] = pd.to_datetime(cpi["DATE"])
    cpi = cpi.rename(columns={"DATE": "date"})

    # Load ISM data
    ism_data_path = os.getenv('ISM_DATA_PATH', '/content/sample_data/AMTMNO.csv')
    ism = pd.read_csv(ism_data_path)
    ism["DATE"] = pd.to_datetime(ism["DATE"])
    ism = ism.rename(columns={"DATE": "date"})

    return df, cpi, ism


def preprocess_data(df, cpi, ism):
    """Complete data preprocessing pipeline."""

    # Process main dataframe
    df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y")
    df['DivYield'] = df['DivYield'].replace('%', '', regex=True)
    df["DivYield"] = pd.to_numeric(df["DivYield"])

    # Merge datasets
    df = pd.merge(df, ism, how="left", on="date")
    df = pd.merge(df, cpi, how="left", on="date")

    # Handle missing values
    df.fillna(method="backfill", inplace=True)
    df.dropna(inplace=True)

    return df


def create_features(df):
    """Create feature matrix and target variable."""
    # Separate features and target
    features_df = df.drop(['date'], axis=1)

    # Define columns to scale (all except target)
    columns_to_scale = [col for col in features_df.columns if col != 'us_5_year_yields']
    df_to_scale = features_df[columns_to_scale]
    df_unscaled = features_df[['us_5_year_yields']]

    # Apply scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_to_scale)
    scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale, index=features_df.index)

    # Combine scaled and unscaled data
    features_df = pd.concat([scaled_df, df_unscaled], axis=1)

    return features_df, scaler


def perform_pca_and_split(features_df, n_components=3, train_ratio=0.8):
    """Perform PCA analysis and train-test split."""
    # Split data
    train_size = int(len(features_df) * train_ratio)
    train_df, test_df = features_df[:train_size], features_df[train_size + 1:]

    # Separate features and target
    y_train = train_df['us_5_year_yields']
    x_train = train_df.drop('us_5_year_yields', axis=1)
    x_test = test_df.drop('us_5_year_yields', axis=1)
    y_test = test_df['us_5_year_yields']

    # Calculate covariance matrix
    cov_matrix = np.cov(x_train, rowvar=False)

    # Apply PCA
    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)  # Use transform, not fit_transform for test

    # Convert to DataFrame
    x_train_pca = pd.DataFrame(x_train_pca, columns=[f'Component {i+1}' for i in range(n_components)])
    x_test_pca = pd.DataFrame(x_test_pca, columns=[f'Component {i+1}' for i in range(n_components)])

    return x_train_pca, x_test_pca, y_train, y_test, pca, cov_matrix


def split_dataframe(df, chunk_size):
    """Split dataframe into chunks of specified size, filtering out small chunks."""
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    filtered_chunks = [chunk for chunk in chunks if len(chunk) >= chunk_size]
    return filtered_chunks


def prepare_sequences(x_train, x_test, y_train, y_test, sequence_length):
    """Convert data into sequences for LSTM training."""

    # Split into sequences
    x_train_seq = split_dataframe(x_train, sequence_length)
    x_test_seq = split_dataframe(x_test, sequence_length)
    y_train_seq = split_dataframe(y_train, sequence_length)
    y_test_seq = split_dataframe(y_test, sequence_length)

    # Convert to numpy arrays
    x_train_np = [np.array(seq) for seq in x_train_seq]
    x_test_np = [np.array(seq) for seq in x_test_seq]
    y_train_np = [np.array(seq) for seq in y_train_seq]
    y_test_np = [np.array(seq) for seq in y_test_seq]

    # Convert to PyTorch tensors
    X_train = torch.tensor(x_train_np).type(torch.float32)
    Y_train = torch.tensor(y_train_np).type(torch.float32)
    X_test = torch.tensor(x_test_np).type(torch.float32)
    Y_test = torch.tensor(y_test_np).type(torch.float32)

    # Transpose for LSTM input format: (seq_len, batch_size, input_size)
    X_train = X_train.transpose(0, 1)
    Y_train = Y_train.transpose(0, 1).reshape(sequence_length, -1, 1)
    X_test = X_test.transpose(0, 1)
    Y_test = Y_test.transpose(0, 1).reshape(sequence_length, -1, 1)

    return X_train, Y_train, X_test, Y_test
