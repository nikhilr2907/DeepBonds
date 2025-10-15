"""
DeepBonds: LSTM Encoder-Decoder for Bond Yield Prediction with Professor Forcing

Main training and prediction script for bond yield forecasting using deep learning.
"""

import os
import argparse
from dotenv import load_dotenv

from src.config import *
from src.data_loader import (
    load_data, preprocess_data, create_features,
    perform_pca_and_split, prepare_sequences
)
from src.models import LSTMSeq2Seq, ModelAlternate
from src.visualization import (
    plot_covariance_matrix, plot_pca_explained_variance,
    plot_feature_loadings, plot_predictions
)
from src.utils import print_metrics

# Load environment variables
load_dotenv()


def main(args):
    """Main training pipeline."""

    print("="*60)
    print("DeepBonds: Bond Yield Prediction with Professor Forcing")
    print("="*60)

    # 1. Load data
    print("\n[1/7] Loading data...")
    df, cpi, ism = load_data()
    print(f"Loaded bond data shape: {df.shape}")
    print(f"Loaded CPI data shape: {cpi.shape}")
    print(f"Loaded ISM data shape: {ism.shape}")

    # 2. Preprocess data
    print("\n[2/7] Preprocessing data...")
    df = preprocess_data(df, cpi, ism)
    features_df, scaler = create_features(df)
    print(f"Preprocessed data shape: {features_df.shape}")
    print(f"Features: {list(features_df.columns)}")

    # 3. Perform PCA and split data
    print("\n[3/7] Performing PCA and splitting data...")
    x_train, x_test, y_train, y_test, pca, cov_matrix = perform_pca_and_split(
        features_df,
        n_components=INPUT_SIZE,
        train_ratio=TRAIN_RATIO
    )
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # 4. Prepare sequences for LSTM
    print("\n[4/7] Preparing sequences for LSTM...")
    X_train, Y_train, X_test, Y_test = prepare_sequences(
        x_train, x_test, y_train, y_test,
        sequence_length=SEQUENCE_LENGTH
    )
    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_test shape: {Y_test.shape}")

    # Move to device
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    # 5. Train base model with teacher forcing
    print("\n[5/7] Training LSTM Seq2Seq model with teacher forcing...")
    model = LSTMSeq2Seq(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
    model = model.to(device)

    avg_loss = model.train_model(
        X_train, Y_train,
        n_epochs=args.epochs,
        target_len=TARGET_LENGTH,
        batch_size=BATCH_SIZE,
        training_prediction='teacher_forcing',
        teacher_forcing_ratio=args.teacher_forcing_ratio,
        learning_rate=args.learning_rate,
        dynamic_tf=args.dynamic_tf
    )
    print(f"Average training loss: {avg_loss:.4f}")

    # 6. Optional: Train with Professor Forcing
    if args.use_professor_forcing:
        print("\n[6/7] Training with Professor Forcing (adversarial)...")
        model_alt = ModelAlternate(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE).to(device)
        model_alt.load_state_dict(torch.load('trained_model.pth'))
        model_alt.adversarial_train(
            input_tensor=X_train,
            target_tensor=Y_train,
            n_epochs=args.pf_epochs,
            target_len=TARGET_LENGTH,
            batch_size=BATCH_SIZE,
            learning_rate=args.pf_learning_rate
        )
        model = model_alt
    else:
        print("\n[6/7] Skipping Professor Forcing (use --use-professor-forcing to enable)")

    # 7. Generate predictions and evaluate
    print("\n[7/7] Generating predictions and evaluating...")
    X_test_cpu = X_test.cpu()
    Y_test_cpu = Y_test.cpu().numpy()

    predictions = model.predict(X_test_cpu, TARGET_LENGTH)
    print("\nTest Set Metrics:")
    metrics = print_metrics(predictions, Y_test_cpu)

    # Visualizations (optional)
    if args.visualize:
        print("\nGenerating visualizations...")

        # Plot covariance matrix
        cov_fig = plot_covariance_matrix(cov_matrix)
        cov_fig.write_html("output_covariance_matrix.html")

        # Plot PCA explained variance
        pca_fig = plot_pca_explained_variance(pca)
        pca_fig.write_html("output_pca_variance.html")

        # Plot predictions
        pred_fig = plot_predictions(Y_test_cpu, predictions)
        pred_fig.write_html("output_predictions.html")

        print("Visualizations saved to HTML files")

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Model saved to: {os.getenv('MODEL_SAVE_PATH', 'trained_model.pth')}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepBonds: Bond Yield Prediction")

    # Training parameters
    parser.add_argument('--epochs', type=int, default=N_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE,
                       help='Learning rate for optimizer')
    parser.add_argument('--teacher-forcing-ratio', type=float, default=0.6,
                       help='Teacher forcing ratio (0.0 to 1.0)')
    parser.add_argument('--dynamic-tf', action='store_true',
                       help='Use dynamic teacher forcing (gradually decrease ratio)')

    # Professor Forcing parameters
    parser.add_argument('--use-professor-forcing', action='store_true',
                       help='Enable Professor Forcing training')
    parser.add_argument('--pf-epochs', type=int, default=500,
                       help='Number of epochs for Professor Forcing')
    parser.add_argument('--pf-learning-rate', type=float, default=0.003,
                       help='Learning rate for Professor Forcing')

    # Visualization
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')

    args = parser.parse_args()
    main(args)
