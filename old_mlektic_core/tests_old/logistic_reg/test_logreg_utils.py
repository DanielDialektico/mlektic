import pytest
import tensorflow as tf
import numpy as np
from mlektic.logistic_reg.logreg_utils import calculate_categorical_crossentropy, calculate_accuracy, calculate_precision, calculate_recall, calculate_f1_score, calculate_confusion_matrix

def test_calculate_categorical_crossentropy():
    y_true = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = tf.constant([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]])
    cce = calculate_categorical_crossentropy(y_true, y_pred).numpy()
    expected_cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred).numpy()
    assert np.allclose(cce, expected_cce), f"Expected {expected_cce}, but got {cce}"

def test_calculate_accuracy():
    y_true = tf.constant([[1, 0], [0, 1], [1, 0], [0, 1]])
    y_pred = tf.constant([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.4, 0.6]])
    accuracy = calculate_accuracy(y_true, y_pred).numpy()
    expected_accuracy = np.mean([1, 1, 1, 1])
    assert np.isclose(accuracy, expected_accuracy), f"Expected {expected_accuracy}, but got {accuracy}"

def test_calculate_precision():
    y_true = tf.constant([[1, 0], [0, 1], [1, 0], [0, 1]])
    y_pred = tf.constant([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.4, 0.6]])
    precision = calculate_precision(y_true, y_pred).numpy()
    expected_precision = 1.0
    assert np.isclose(precision, expected_precision), f"Expected {expected_precision}, but got {precision}"

def test_calculate_recall():
    y_true = tf.constant([[1, 0], [0, 1], [1, 0], [0, 1]])
    y_pred = tf.constant([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.4, 0.6]])
    recall = calculate_recall(y_true, y_pred).numpy()
    expected_recall = 1.0
    assert np.isclose(recall, expected_recall), f"Expected {expected_recall}, but got {recall}"

def test_calculate_f1_score():
    y_true = tf.constant([[1, 0], [0, 1], [1, 0], [0, 1]])
    y_pred = tf.constant([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.4, 0.6]])
    f1_score = calculate_f1_score(y_true, y_pred).numpy()
    expected_f1_score = 1.0
    assert np.isclose(f1_score, expected_f1_score), f"Expected {expected_f1_score}, but got {f1_score}"

def test_calculate_confusion_matrix():
    y_true = tf.constant([[1, 0], [0, 1], [1, 0], [0, 1]])
    y_pred = tf.constant([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.4, 0.6]])
    confusion_matrix = calculate_confusion_matrix(y_true, y_pred).numpy()
    expected_confusion_matrix = np.array([[2, 0], [0, 2]])
    assert np.array_equal(confusion_matrix, expected_confusion_matrix), f"Expected {expected_confusion_matrix}, but got {confusion_matrix}"

if __name__ == "__main__":
    pytest.main()