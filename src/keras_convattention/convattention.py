# src/keras_convattention/convattention.py

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Lambda, Dot, Activation, Concatenate, Layer, RepeatVector, Add
from .attention import Attention  # Import the Attention class from attention.py

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Layer, Add
from tensorflow.keras import initializers
import numpy as np
import random
import os

# Set random seed for reproducibility
seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

# Import the Attention class from attention.py
from .attention import Attention  
# Set TensorFlow to use deterministic algorithms
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# class ConvAttentionLayer(Layer):
#     SCORE_LUONG = 'luong'
#     SCORE_BAHDANAU = 'bahdanau'

#     def __init__(self, filters=64, kernel_size=3, pool_size=2, units=128, score='bahdanau', **kwargs):
#         super(ConvAttentionLayer, self).__init__(**kwargs)
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.pool_size = pool_size
#         self.units = units
#         self.score = score
#         self.activation = activation  
#         # CNN and MaxPooling Layers
#         self.conv1d = Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation=self.activation, padding='same', name='cnn_layer')
#         self.dense_projection = Dense(self.d_units, activation='relu', name='dense_projection')
#         self.add_layer = Add()
#         # Attention layer (using the Attention from attention.py)
#         self.attention = Attention(units=self.units, score=self.score, name='attention_layer')

#     def build(self, input_shape):
#         super(ConvAttentionLayer, self).build(input_shape)

#     def call(self, inputs, training=None, **kwargs):
#         # Apply CNN and pooling layers
#         x = self.conv1d(inputs)
#         # x = self.pool1d(x)

#         self.d_units=inputs.shape[-1]
#         x = self.dense_projection(x)
#         x = self.add_layer([inputs,x])
#         # Pass through the attention layer (training argument passed implicitly)
#         return self.attention(x, training=training)

#     def compute_output_shape(self, input_shape):
#         return input_shape[0], self.units

#     def get_config(self):
#         config = super(ConvAttentionLayer, self).get_config()
#         config.update({
#             'filters': self.filters,
#             'kernel_size': self.kernel_size,
#             'pool_size': self.pool_size,
#             'units': self.units,
#             'score': self.score,
#             'activation': self.activation
#         })
#         return config
# class ConvAttentionLayer(Layer):
#     SCORE_LUONG = 'luong'
#     SCORE_BAHDANAU = 'bahdanau'

#     def __init__(self, filters=64, kernel_size=3, pool_size=2, units=128, score='bahdanau', activation='relu', **kwargs):
#         super(ConvAttentionLayer, self).__init__(**kwargs)
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.pool_size = pool_size
#         self.units = units
#         self.score = score
#         self.activation = activation  # Set the default activation to 'relu'
#         # CNN and MaxPooling Layers
#         self.conv1d = Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation=self.activation, padding='same', name='cnn_layer')
#         self.dense_projection = Dense(self.d_units, activation='relu', name='dense_projection')
#         self.add_layer = Add()
#         # Attention layer (using the Attention from attention.py)
#         self.attention = Attention(units=self.units, score=self.score, name='attention_layer')

#     def build(self, input_shape):
#         super(ConvAttentionLayer, self).build(input_shape)

#     def call(self, inputs, training=None, **kwargs):
#         # Apply CNN and pooling layers
#         x = self.conv1d(inputs)
#         self.d_units = inputs.shape[-1]
#         print(self.d_units)
#         x = self.dense_projection(x)
#         x = self.add_layer([inputs, x])
#         # Pass through the attention layer (training argument passed implicitly)
#         return self.attention(x, training=training)

#     def compute_output_shape(self, input_shape):
#         return input_shape[0], self.units

#     def get_config(self):
#         config = super(ConvAttentionLayer, self).get_config()
#         config.update({
#             'filters': self.filters,
#             'kernel_size': self.kernel_size,
#             'pool_size': self.pool_size,
#             'units': self.units,
#             'score': self.score,
#             'activation': self.activation
#         })
#         return config
class ConvAttentionLayer(Layer):
    SCORE_LUONG = 'luong'
    SCORE_BAHDANAU = 'bahdanau'

    def __init__(self, filters=64, kernel_size=3, pool_size=2, units=128, score='bahdanau', activation='relu', **kwargs):
        super(ConvAttentionLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.units = units
        self.score = score
        self.activation = activation  # Set the default activation to 'relu'
        # CNN and MaxPooling Layers
        self.conv1d = Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation=self.activation, padding='same', name='cnn_layer')
        self.add_layer = Add()
        # Attention layer (using the Attention from attention.py)
        self.attention = Attention(units=self.units, score=self.score, name='attention_layer')

    def build(self, input_shape):
        # Set the dynamic units for dense_projection based on the input shape
        self.d_units = input_shape[-1]  # The number of input channels or features
        self.dense_projection = Dense(self.d_units, activation='relu', name='dense_projection')  # Use dynamic units here
        super(ConvAttentionLayer, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        # Apply CNN layer
        x = self.conv1d(inputs)
        # Apply the dynamic dense_projection layer after the convolution
        x = self.dense_projection(x)
        x = self.add_layer([inputs, x])  # Add the original inputs to the output of dense_projection
        # Pass through the attention layer
        return self.attention(x, training=training)

    def compute_output_shape(self, input_shape):
        # Return output shape considering the attention mechanism
        return input_shape[0], self.units

    def get_config(self):
        config = super(ConvAttentionLayer, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'pool_size': self.pool_size,
            'units': self.units,
            'score': self.score,
            'activation': self.activation
        })
        return config
