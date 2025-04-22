# src/keras_convattention/convattention.py

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Lambda, Dot, Activation, Concatenate, Layer, RepeatVector, Add
from .attention import Attention  # Import the Attention class from attention.py

class ConvAttentionLayer(Layer):
    SCORE_LUONG = 'luong'
    SCORE_BAHDANAU = 'bahdanau'

    def __init__(self, filters=64, kernel_size=3, pool_size=2, units=128, score='bahdanau', **kwargs):
        super(ConvAttentionLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.units = units
        self.score = score

        # CNN and MaxPooling Layers
        self.conv1d = Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', padding='same', name='cnn_layer')
        self.pool1d = MaxPooling1D(pool_size=self.pool_size, name='pool_layer')
        self.dense_projection = Dense(self.units, activation='relu', name='dense_projection')

        # Attention layer (using the Attention from attention.py)
        self.attention = Attention(units=self.units, score=self.score, name='attention_layer')

    def build(self, input_shape):
        super(ConvAttentionLayer, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        # Apply CNN and pooling layers
        x = self.conv1d(inputs)
        x = self.pool1d(x)

        # Dense projection layer to ensure compatibility with attention
        x = self.dense_projection(x)

        # Pass through the attention layer (training argument passed implicitly)
        return self.attention(x, training=training)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def get_config(self):
        config = super(ConvAttentionLayer, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'pool_size': self.pool_size,
            'units': self.units,
            'score': self.score
        })
        return config
