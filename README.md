# Keras-ConvAttention

**Keras-ConvAttention** is a Python package built on top of **Keras** and **TensorFlow** that introduces a custom attention mechanism designed specifically for **time series forecasting** tasks. The package extends the traditional attention mechanism by integrating **Convolutional Neural Networks (CNN)**, making it a powerful tool for improving the performance of time series prediction models. 

### Types of Attention Mechanisms

You can use the following types of attention mechanisms in **Keras-ConvAttention**:

- **Luong's Multiplicative Attention**: This attention mechanism uses a dot product to compute attention scores, and is suitable for tasks where interactions between different time steps are captured via a simple multiplicative form of attention.
  
- **Bahdanau's Additive Attention**: This mechanism uses a feedforward neural network to compute attention scores, and is often more expressive than Luong's approach. It adds a layer of flexibility by considering the hidden state and the input sequence in a more non-linear way.

Both of these attention mechanisms are available and can be chosen based on the type of data and forecasting problem you are working on.

---

### Installation

You can install the **keras-convattention** package via **pip**:

```bash
pip install keras-convattention
```
# How It Works

This package contains two main layers:

- **Attention Layer**: Implements Bahdanau and Luong attention mechanisms for time series data.
- **ConvAttention Layer**: Combines CNNs with attention to enhance performance, particularly useful for time series forecasting.

---

### Attention Layer

The **Attention Layer** allows the model to focus on important parts of the input sequence. It supports two types of attention mechanisms:

1. **Luong’s Multiplicative Attention**: Computes attention scores using a dot product.
2. **Bahdanau’s Additive Attention**: Computes attention scores using a feedforward neural network.

#### Pseudo Code for Attention Layer

1. **Input**: Sequence of vectors (hidden states of an RNN or LSTM layer).
2. **Luong's Attention**:
   - Compute attention scores using the dot product of the hidden states and a learned weight matrix.
   - Apply softmax on the scores to get attention weights.
   - Calculate context vector as the weighted sum of the input sequence.
3. **Bahdanau's Attention**:
   - Compute attention scores by applying two learned weight matrices to the hidden states.
   - Use a non-linear activation (tanh) on the sum of these weighted hidden states.
   - Apply softmax to get attention weights.
   - Compute the context vector using the attention weights.

**Output**: Context vector (a weighted sum of the input sequence based on attention).

---

### ConvAttention Layer

The **ConvAttention Layer** integrates a **Convolutional Neural Network (CNN)** with the **Attention Layer** to extract spatial features and improve the performance of the attention mechanism, making it especially effective for time series forecasting.

#### Pseudo Code for ConvAttention Layer

1. **Input**: Sequence of vectors (hidden states or input features).
2. **Convolutional Layer**:
   - Apply 1D Convolutional layer to the input sequence to extract local features.
   - Optionally apply dilation or other operations to capture temporal dependencies.
3. **Max Pooling**:
   - Apply max pooling to downsample the feature map, retaining the most important features.
4. **Dense Projection**:
   - Apply a dense layer to transform the pooled features into a desired size for attention.
5. **Attention Mechanism**:
   - Apply the Attention Layer (Luong or Bahdanau) to focus on important parts of the transformed feature sequence.
6. **Output**: The final weighted sum of the input sequence, adjusted based on learned attention weights.

**Output**: A context vector combined with the attention-enhanced features.

---

### Model Example with LSTM

Here is an example of how you can integrate the **Attention Layer** and **ConvAttention Layer** with an **LSTM** model for time series forecasting:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from keras_convattention.attention import Attention
from keras_convattention.convattention import ConvAttentionLayer

# Create a base LSTM layer
def clone_base_lstm(input_shape):
    lstm_layer = LSTM(512, return_sequences=True, input_shape=input_shape)
    return clone_model(lstm_layer)  # Clone the LSTM layer

# Model definitions using LSTM + CNN + Attention
def build_lstm_cnn_attention_model(input_shape):
    model = Sequential([
        clone_base_lstm(input_shape),  # Cloned LSTM (trainable)
        ConvAttentionLayer(filters=32, kernel_size=1, units=128, score='bahdanau', activation='sigmoid'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')
    return model
```
### Model Description

This model uses **LSTM** to handle sequential data, while the **Attention** and **ConvAttention** layers enhance the model’s ability to focus on the most relevant parts of the input sequence. The **LSTM** is responsible for learning long-term dependencies in the data, while the **Attention Layer** helps the model concentrate on important time steps, and the **ConvAttention Layer** further boosts performance by applying **Convolutional Neural Networks (CNN)** along with the attention mechanism.

---

### Hyperparameter Optimization with Optuna

You can fine-tune the hyperparameters of your model using **Optuna**, which helps to identify the best set of hyperparameters for your task. This allows you to optimize key parameters like the number of filters, kernel size, and attention units to improve the model’s performance.

To run hyperparameter optimization, execute the following script:

```bash
python ./example/optimize.py
```

### Benchmarking Time Series Forecasting

For benchmarking the **ConvAttention Layer** on a time series dataset, you can run the following script:

```bash
python ./example/benchmark.py
```
