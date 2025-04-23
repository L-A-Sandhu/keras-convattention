# import tensorflow as tf
# from tensorflow.keras import backend as K
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Layer, Add
# from tensorflow.keras import initializers
# import numpy as np
# import random
# import os

# # Set random seed for reproducibility
# seed_value = 42
# tf.random.set_seed(seed_value)
# np.random.seed(seed_value)
# random.seed(seed_value)

# # Set TensorFlow to use deterministic algorithms (in case of GPU operations)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'

# # Import the Attention class from attention.py
# from .attention import Attention  

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

#         # CNN and MaxPooling Layers with specified initializers for reproducibility
#         self.conv1d = Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', 
#                              padding='same', name='cnn_layer', kernel_initializer=initializers.GlorotUniform(seed=seed_value))
#         self.pool1d = MaxPooling1D(pool_size=self.pool_size, name='pool_layer')
#         self.dense_projection = Dense(self.units, activation='relu', name='dense_projection',
#                                       kernel_initializer=initializers.GlorotUniform(seed=seed_value))

#         # Attention layer (using the Attention from attention.py)
#         self.attention = Attention(units=self.units, score=self.score, name='attention_layer')

#     def build(self, input_shape):
#         super(ConvAttentionLayer, self).build(input_shape)

#     def call(self, inputs, training=None, **kwargs):
#         # Apply CNN and pooling layers
#         x = self.conv1d(inputs)
#         x = self.pool1d(x)

#         # Dense projection layer to ensure compatibility with attention
#         x = self.dense_projection(x)

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
#             'score': self.score
#         })
#         return config
# import tensorflow as tf
# from tensorflow.keras import backend as K
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Layer, Add
# from tensorflow.keras import initializers
# import numpy as np
# import random
# import os

# # Set random seed for reproducibility
# seed_value = 42
# tf.random.set_seed(seed_value)
# np.random.seed(seed_value)
# random.seed(seed_value)

# # Import the Attention class from attention.py
# from .attention import Attention  
# # Set TensorFlow to use deterministic algorithms
# os.environ['TF_DETERMINISTIC_OPS'] = '1'

# class ConvAttentionLayer(Layer):
#     SCORE_LUONG = 'luong'
#     SCORE_BAHDANAU = 'bahdanau'

#     def __init__(self, filters=64, kernel_size=3, pool_size=2, units=128, 
#                  score='bahdanau', dilation_rate=1, padding='same', 
#                  activation='relu', **kwargs):
#         super(ConvAttentionLayer, self).__init__(**kwargs)
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.pool_size = pool_size
#         self.units = units
#         self.score = score
#         self.dilation_rate = dilation_rate
#         self.padding = padding
#         self.activation = activation

#         # Enhanced CNN layer with additional arguments
#         self.conv1d = Conv1D(
#             filters=self.filters,
#             kernel_size=self.kernel_size,
#             activation=self.activation,
#             padding=self.padding,
#             dilation_rate=self.dilation_rate,
#             kernel_initializer=initializers.GlorotUniform(seed=seed_value),
#             bias_initializer=initializers.Zeros(),
#             name='cnn_layer'
#         )
        
#         self.pool1d = MaxPooling1D(
#             pool_size=self.pool_size, 
#             name='pool_layer'
#         )
        
#         self.dense_projection = Dense(
#             self.units, 
#             activation='relu',
#             kernel_initializer=initializers.GlorotUniform(seed=seed_value),
#             bias_initializer=initializers.Zeros(),
#             name='dense_projection'
#         )

#         self.attention = Attention(
#             units=self.units, 
#             score=self.score, 
#             name='attention_layer'
#         )

#     def build(self, input_shape):
#         super(ConvAttentionLayer, self).build(input_shape)

#     def call(self, inputs, training=None, **kwargs):
#         x = self.conv1d(inputs)
#         x = self.pool1d(x)
#         x = self.dense_projection(x)
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
#             'dilation_rate': self.dilation_rate,
#             'padding': self.padding,
#             'activation': self.activation
#         })
#         return config
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

class ConvAttentionLayer(Layer):
    SCORE_LUONG = 'luong'
    SCORE_BAHDANAU = 'bahdanau'

    def __init__(self, filters=64, kernel_size=3, pool_size=2, units=128, 
                 score='bahdanau', dilation_rate=1, padding='same', 
                 activation='relu', **kwargs):
        super(ConvAttentionLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.units = units
        self.score = score
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.activation = activation

        # Enhanced CNN layer with additional arguments
        self.conv1d = Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            activation=self.activation,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            kernel_initializer=initializers.GlorotUniform(seed=seed_value),
            bias_initializer=initializers.Zeros(),
            name='cnn_layer'
        )
        
        self.pool1d = MaxPooling1D(
            pool_size=self.pool_size, 
            name='pool_layer'
        )
        
        # Change the Dense layer's units to be equal to num_features
        self.dense_projection = Dense(
            self.filters,  # Set units to filters (which is equivalent to num_features)
            activation='relu',
            kernel_initializer=initializers.GlorotUniform(seed=seed_value),
            bias_initializer=initializers.Zeros(),
            name='dense_projection'
        )

        self.attention = Attention(
            units=self.units, 
            score=self.score, 
            name='attention_layer'
        )

    def build(self, input_shape):
        super(ConvAttentionLayer, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1d(inputs)
        x = self.pool1d(x)
        x = self.dense_projection(x)
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
            'score': self.score,
            'dilation_rate': self.dilation_rate,
            'padding': self.padding,
            'activation': self.activation
        })
        return config
