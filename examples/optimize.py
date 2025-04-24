import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from keras_convattention.convattention import ConvAttentionLayer
from keras_convattention.attention import Attention
import timemesh as tm
from tensorflow.keras.optimizers import Adam
import optuna

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set TensorFlow to use deterministic operations
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Load and prepare data
df = pd.read_csv("data.csv")
input_cols = [
    "C_WD50M", "C_WS50M", "C_PS", "C_T2M", "C_QV2M",
    "N_WD50M", "N_WS50M", "N_PS", "N_T2M", "N_QV2M",
    "S_WD50M", "S_WS50M", "S_PS", "S_T2M", "S_QV2M",
    "E_WD50M", "E_WS50M", "E_PS", "E_T2M", "E_QV2M", 
    "W_WD50M", "W_WS50M", "W_PS", "W_T2M", "W_QV2M", 
    "NE_WD50M", "NE_WS50M", "NE_PS", "NE_T2M", "NE_QV2M",
    "NW_WD50M", "NW_WS50M", "NW_PS", "NW_T2M", "NW_QV2M",
    "SE_WD50M", "SE_WS50M", "SE_PS", "SE_T2M", "SE_QV2M",
    "SW_WD50M", "SW_WS50M", "SW_PS", "SW_T2M", "SW_QV2M"
]

output_cols = ["C_WS50M"]

print("\n--- Data Preparation ---")
loader_norm = tm.DataLoader(T=24, H=3, input_cols=input_cols, output_cols=output_cols, 
                           norm="Z", step=6, ratio={'train': 70, 'test': 15, 'valid': 15})
X_train, Y_train, X_test, Y_test, X_valid, Y_valid, input_params_z, output_params_z = loader_norm.load_csv("data.csv")

# Model definitions with separate components
from tensorflow.keras.models import clone_model

# Create a base LSTM layer with fixed initialization
def create_base_lstm(input_shape):
    return LSTM(
        512,
        return_sequences=True,
        input_shape=input_shape,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),  # Fixed seed
        recurrent_initializer=tf.keras.initializers.Orthogonal(seed=42),   # Fixed seed
        bias_initializer='zeros'
    )

# Function to clone the base LSTM and ensure it's trainable (for both models)
def clone_base_lstm(input_shape):
    lstm_layer = create_base_lstm(input_shape)
    return clone_model(lstm_layer)  # This will clone the LSTM layer

# Model definitions using cloned LSTM
def build_lstm_attention_model(input_shape):
    model = Sequential([
        clone_base_lstm(input_shape),  # Cloned LSTM (trainable)
        Attention(units=128, score='bahdanau'),
        Dense(3)
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')
    return model

def build_lstm_cnn_attention_model(input_shape, filters, kernel_size, activation):
    model = Sequential([
        clone_base_lstm(input_shape),  # Cloned LSTM (trainable)
        ConvAttentionLayer(filters=filters, kernel_size=kernel_size, units=128, score='bahdanau', activation=activation),
        Dense(3)
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')
    return model

# Objective function for Optuna
def objective(trial):
    # Hyperparameters to tune
    filters = trial.suggest_categorical('filters', [8, 16, 32, 48, 64,  128])
    kernel_size = trial.suggest_categorical('kernel_size', [1,2,3,4, 5])
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid', 'swish'])

    # Initialize model with the selected hyperparameters
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_cnn_attention_model(input_shape, filters, kernel_size, activation)

    # Early stopping
    early_stopping = EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True, verbose=1)

    # Train the model
    history = model.fit(
        X_train, Y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_valid, Y_valid),
        callbacks=[early_stopping],
        verbose=0
    )

    # Predict and evaluate
    preds = model.predict(X_test)
    preds = tm.Normalizer.denormalize(preds, params=output_params_z, method="Z", feature_order=output_cols)
    Y_test_denorm = tm.Normalizer.denormalize(Y_test, params=output_params_z, method="Z", feature_order=output_cols).squeeze()

    mse = mean_squared_error(Y_test_denorm, preds)
    return mse  # Optuna will minimize the MSE

# Study optimization using Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)  # You can increase the number of trials for better results

# Get the best trial result
best_trial = study.best_trial
print(f"\nBest Trial: {best_trial.params}")
print(f"Best MSE: {best_trial.value}")
