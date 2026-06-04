import pytest
import tensorflow as tf
import numpy as np
from mlektic.methods.base.regularizers import Regularizers

def test_l1_regularizer():
    lambda_value = 0.01
    regularizer = Regularizers.l1(lambda_value)
    weights = tf.constant([[1.0, -1.0], [2.0, -2.0]], dtype=tf.float32)
    expected_loss = lambda_value * tf.reduce_sum(tf.abs(weights), axis=0).numpy()
    calculated_loss = regularizer(weights).numpy()
    assert np.allclose(calculated_loss, expected_loss), f"Expected {expected_loss}, but got {calculated_loss}"

def test_l2_regularizer():
    lambda_value = 0.01
    regularizer = Regularizers.l2(lambda_value)
    weights = tf.constant([[1.0, -1.0], [2.0, -2.0]], dtype=tf.float32)
    expected_loss = lambda_value * tf.reduce_sum(tf.square(weights), axis=0).numpy()
    calculated_loss = regularizer(weights).numpy()
    assert np.allclose(calculated_loss, expected_loss), f"Expected {expected_loss}, but got {calculated_loss}"

def test_elastic_net_regularizer():
    lambda_value = 0.01
    alpha = 0.5
    regularizer = Regularizers.elastic_net(lambda_value, alpha)
    weights = tf.constant([[1.0, -1.0], [2.0, -2.0]], dtype=tf.float32)
    l1_loss = lambda_value * alpha * tf.reduce_sum(tf.abs(weights), axis=0).numpy()
    l2_loss = lambda_value * (1 - alpha) * tf.reduce_sum(tf.square(weights), axis=0).numpy()
    expected_loss = l1_loss + l2_loss
    calculated_loss = regularizer(weights).numpy()
    assert np.allclose(calculated_loss, expected_loss), f"Expected {expected_loss}, but got {calculated_loss}"

if __name__ == "__main__":
    pytest.main()