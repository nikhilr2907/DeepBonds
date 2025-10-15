"""Visualization utilities for bond yield prediction"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random


def random_color():
    """Generate random RGB color."""
    return f'rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})'


def plot_covariance_matrix(cov_matrix):
    """Plot covariance matrix heatmap."""
    cov_matrix_df = pd.DataFrame(cov_matrix)

    fig = go.Figure(data=go.Heatmap(
        z=cov_matrix_df.values,
        x=list(range(cov_matrix_df.shape[1])),
        y=list(range(cov_matrix_df.shape[0])),
        colorscale='Viridis',
        colorbar=dict(title='Covariance'),
    ))

    fig.update_layout(
        title='Feature Covariance Matrix Heatmap',
        xaxis_title='Feature Index',
        yaxis_title='Feature Index',
        template='plotly_white'
    )

    return fig


def plot_pca_explained_variance(pca):
    """Plot PCA explained variance."""
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    fig = go.Figure()

    # Bar plot for explained variance
    fig.add_trace(go.Bar(
        x=[f'PC{i+1}' for i in range(len(explained_variance))],
        y=explained_variance,
        name='Explained Variance',
        yaxis='y'
    ))

    # Line plot for cumulative explained variance
    fig.add_trace(go.Scatter(
        x=[f'PC{i+1}' for i in range(len(cumulative_variance))],
        y=cumulative_variance,
        mode='lines+markers',
        name='Cumulative Explained Variance',
        yaxis='y2'
    ))

    fig.update_layout(
        title='PCA Explained Variance Analysis',
        xaxis_title='Principal Components',
        yaxis=dict(title='Explained Variance', side='left'),
        yaxis2=dict(title='Cumulative Variance', side='right', overlaying='y'),
        template='plotly_white'
    )

    return fig


def plot_feature_loadings(pca, feature_names):
    """Plot feature loadings for first principal component."""
    loadings = pca.components_[0]
    loadings_df = pd.DataFrame(loadings, index=feature_names, columns=['Loading'])

    # Generate colors
    colors = [random_color() for _ in feature_names]

    fig = go.Figure(data=go.Bar(
        x=loadings_df.index,
        y=loadings_df['Loading'],
        marker_color=colors,
        name='Feature Loadings'
    ))

    fig.update_layout(
        title='Feature Loadings for First Principal Component',
        xaxis_title='Features',
        yaxis_title='Loading Weight',
        template='plotly_white',
        xaxis_tickangle=-45
    )

    return fig


def plot_predictions(y_true, y_pred, title="Bond Yield Predictions"):
    """Plot actual vs predicted bond yields."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=y_true.flatten(),
        mode='lines',
        name='Actual',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        y=y_pred.flatten(),
        mode='lines',
        name='Predicted',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Time Step',
        yaxis_title='Bond Yield',
        template='plotly_white'
    )

    return fig
