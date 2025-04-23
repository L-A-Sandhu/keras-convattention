import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from keras_convattention.convattention import ConvAttentionLayer
from keras_convattention.attention import Attention
import timemesh as tm
from tensorflow.keras.optimizers import Adam

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set TensorFlow to use deterministic operations
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, False)
            tf.config.set_visible_devices(gpu, 'GPU')
    except RuntimeError as e:
        print(e)

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

def build_lstm_cnn_attention_model(input_shape):
    model = Sequential([
        clone_base_lstm(input_shape),  # Cloned LSTM (trainable)
        ConvAttentionLayer(filters=32, kernel_size=1, pool_size=1, units=128, score='bahdanau', dilation_rate=1, padding='causal', activation='sigmoid'),
        Dense(3)
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')
    return model

# Training setup
checkpoint_dir = './checkpoint'
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize models with separate weights
input_shape = (X_train.shape[1], X_train.shape[2])
lstm_attention_model = build_lstm_attention_model(input_shape)
lstm_cnn_attention_model = build_lstm_cnn_attention_model(input_shape)

# Model checkpoints with different file names for each model
checkpoint_lstm_attention = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'lstm_attention_best_weights_{epoch:02d}.keras'),
    save_best_only=True,
    monitor='val_loss',
    verbose=1
)

checkpoint_lstm_cnn_attention = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'lstm_cnn_attention_best_weights_{epoch:02d}.keras'),
    save_best_only=True,
    monitor='val_loss',
    verbose=1
)

# Early stopping
early_stopping = EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True, verbose=1)

# Train models independently
print("\n--- Training LSTM+Attention Model ---")
history_lstm_attention = lstm_attention_model.fit(
    X_train, Y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_valid, Y_valid),
    callbacks=[checkpoint_lstm_attention, early_stopping],
    verbose=1
)

print("\n--- Training LSTM+CNN+Attention Model ---")
history_lstm_cnn_attention = lstm_cnn_attention_model.fit(
    X_train, Y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_valid, Y_valid),
    callbacks=[checkpoint_lstm_cnn_attention, early_stopping],
    verbose=1
)

# Save model architecture as JSON
with open(os.path.join(checkpoint_dir, 'lstm_attention_model.json'), 'w') as json_file:
    json_file.write(lstm_attention_model.to_json())

with open(os.path.join(checkpoint_dir, 'lstm_cnn_attention_model.json'), 'w') as json_file:
    json_file.write(lstm_cnn_attention_model.to_json())

# Plotting the training and validation loss for both models
plt.figure(figsize=(12, 6))

# LSTM + Attention Training Plot
plt.subplot(1, 2, 1)
plt.plot(history_lstm_attention.history['loss'], label='Training Loss')
plt.plot(history_lstm_attention.history['val_loss'], label='Validation Loss')
plt.title('LSTM + Attention Model Training History')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# LSTM + CNN + Attention Training Plot
plt.subplot(1, 2, 2)
plt.plot(history_lstm_cnn_attention.history['loss'], label='Training Loss')
plt.plot(history_lstm_cnn_attention.history['val_loss'], label='Validation Loss')
plt.title('LSTM + CNN + Attention Model Training History')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Save the plot
plt.tight_layout()
plt.savefig(os.path.join(checkpoint_dir, 'training_plots.png'))

# Predict using both models
lstm_attention_preds = lstm_attention_model.predict(X_test)
lstm_attention_preds = tm.Normalizer.denormalize(lstm_attention_preds, params=output_params_z, method="Z", feature_order=output_cols)

lstm_cnn_attention_preds = lstm_cnn_attention_model.predict(X_test)
lstm_cnn_attention_preds = tm.Normalizer.denormalize(lstm_cnn_attention_preds, params=output_params_z, method="Z", feature_order=output_cols)

# Inverse transform predictions and actual values to get them back to original scale
Y_test = tm.Normalizer.denormalize(Y_test, params=output_params_z, method="Z", feature_order=output_cols).squeeze()

print("Shape of lstm_attention_preds:", np.shape(lstm_attention_preds))
print("Shape of lstm_cnn_attention_preds:", np.shape(lstm_cnn_attention_preds))
print("Shape of Y_test:", np.shape(Y_test))

lstm_attention_mse = mean_squared_error(Y_test, lstm_attention_preds)
lstm_cnn_attention_mse = mean_squared_error(Y_test, lstm_cnn_attention_preds)

lstm_attention_mae = mean_absolute_error(Y_test, lstm_attention_preds)
lstm_cnn_attention_mae = mean_absolute_error(Y_test, lstm_cnn_attention_preds)

lstm_attention_r2 = r2_score(Y_test, lstm_attention_preds)
lstm_cnn_attention_r2 = r2_score(Y_test, lstm_cnn_attention_preds)

# Print the results
print(f"LSTM + Attention Model MSE: {lstm_attention_mse:.4f}, MAE: {lstm_attention_mae:.4f}, R²: {lstm_attention_r2:.4f}")
print(f"LSTM + CNN + Attention Model MSE: {lstm_cnn_attention_mse:.4f}, MAE: {lstm_cnn_attention_mae:.4f}, R²: {lstm_cnn_attention_r2:.4f}")

# Plot the predictions of both models against actual values
plt.figure(figsize=(10, 6))
plt.plot(Y_test, label='Actual', color='black')
plt.plot(lstm_attention_preds, label='LSTM + Attention Predictions', color='blue', linestyle='--')
plt.plot(lstm_cnn_attention_preds, label='LSTM + CNN + Attention Predictions', color='red', linestyle='--')
plt.title('Model Comparison: LSTM + Attention vs LSTM + CNN + Attention')
plt.xlabel('Time')
plt.ylabel('Number of Passengers')
plt.legend()
plt.savefig(os.path.join(checkpoint_dir, 'model_comparison_plot.png'))
