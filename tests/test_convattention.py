# tests/test_convattention.py

import pytest
import tensorflow as tf
from keras_convattention.convattention import ConvAttentionLayer

@pytest.fixture
def sample_data():
    return tf.random.normal((32, 12, 1))  # Example shape: (batch_size, time_steps, features)

def test_conv_attention_layer(sample_data):
    layer = ConvAttentionLayer(units=128)  # Example: units=128
    output = layer(sample_data)
    assert output.shape == (32, 128)  # Adjust based on your expected output
