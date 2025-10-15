"""Utility functions for DeepBonds"""

import numpy as np
import torch


def list_to_numpy(list_of_lists):
    """Convert all lists in a list of lists to NumPy arrays."""
    collection = [np.array(sublist) for sublist in list_of_lists]
    return collection


def numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest):
    """Convert numpy array to PyTorch tensor."""
    X_train_torch = torch.tensor(Xtrain).type(torch.float32)
    Y_train_torch = torch.tensor(Ytrain).type(torch.float32)
    X_test_torch = torch.tensor(Xtest).type(torch.float32)
    Y_test_torch = torch.tensor(Ytest).type(torch.float32)

    return X_train_torch, Y_train_torch, X_test_torch, Y_test_torch


def calculate_mse(predictions, targets):
    """Calculate Mean Squared Error."""
    return np.mean((predictions - targets) ** 2)


def calculate_rmse(predictions, targets):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(calculate_mse(predictions, targets))


def calculate_mae(predictions, targets):
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(predictions - targets))


def print_metrics(predictions, targets):
    """Print evaluation metrics."""
    mse = calculate_mse(predictions, targets)
    rmse = calculate_rmse(predictions, targets)
    mae = calculate_mae(predictions, targets)

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    return {'mse': mse, 'rmse': rmse, 'mae': mae}
