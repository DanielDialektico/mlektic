import pytest
import tensorflow as tf
import numpy as np
from mlektic.linear_reg.linreg_utils import calculate_mse, calculate_rmse, calculate_mae, calculate_mape, calculate_r2, calculate_pearson_correlation

def test_calculate_mse():
    y_true = tf.constant([1.0, 2.0, 3.0])
    y_pred = tf.constant([1.1, 1.9, 3.2])
    mse = calculate_mse(y_true, y_pred).numpy()
    expected_mse = np.mean((np.array([1.1, 1.9, 3.2]) - np.array([1.0, 2.0, 3.0]))**2)
    assert np.isclose(mse, expected_mse), f"Expected {expected_mse}, but got {mse}"

def test_calculate_rmse():
    y_true = tf.constant([1.0, 2.0, 3.0])
    y_pred = tf.constant([1.1, 1.9, 3.2])
    rmse = calculate_rmse(y_true, y_pred).numpy()
    expected_rmse = np.sqrt(np.mean((np.array([1.1, 1.9, 3.2]) - np.array([1.0, 2.0, 3.0]))**2))
    assert np.isclose(rmse, expected_rmse), f"Expected {expected_rmse}, but got {rmse}"

def test_calculate_mae():
    y_true = tf.constant([1.0, 2.0, 3.0])
    y_pred = tf.constant([1.1, 1.9, 3.2])
    mae = calculate_mae(y_true, y_pred).numpy()
    expected_mae = np.mean(np.abs(np.array([1.1, 1.9, 3.2]) - np.array([1.0, 2.0, 3.0])))
    assert np.isclose(mae, expected_mae), f"Expected {expected_mae}, but got {mae}"

def test_calculate_mape():
    y_true = tf.constant([1.0, 2.0, 3.0])
    y_pred = tf.constant([1.1, 1.9, 3.2])
    mape = calculate_mape(y_true, y_pred).numpy()
    expected_mape = np.mean(np.abs((np.array([1.0, 2.0, 3.0]) - np.array([1.1, 1.9, 3.2])) / np.array([1.0, 2.0, 3.0]))) * 100
    assert np.isclose(mape, expected_mape), f"Expected {expected_mape}, but got {mape}"

def test_calculate_r2():
    y_true = tf.constant([1.0, 2.0, 3.0])
    y_pred = tf.constant([1.1, 1.9, 3.2])
    r2 = calculate_r2(y_true, y_pred).numpy()
    ss_total = np.sum((np.array([1.0, 2.0, 3.0]) - np.mean(np.array([1.0, 2.0, 3.0])))**2)
    ss_residual = np.sum((np.array([1.0, 2.0, 3.0]) - np.array([1.1, 1.9, 3.2]))**2)
    expected_r2 = 1 - (ss_residual / ss_total)
    assert np.isclose(r2, expected_r2), f"Expected {expected_r2}, but got {r2}"

def test_calculate_pearson_correlation():
    y_true = tf.constant([1.0, 2.0, 3.0])
    y_pred = tf.constant([1.1, 1.9, 3.2])
    pearson_corr = calculate_pearson_correlation(y_true, y_pred).numpy()
    mean_y_true = np.mean(np.array([1.0, 2.0, 3.0]))
    mean_y_pred = np.mean(np.array([1.1, 1.9, 3.2]))
    covariance = np.sum((np.array([1.0, 2.0, 3.0]) - mean_y_true) * (np.array([1.1, 1.9, 3.2]) - mean_y_pred))
    std_y_true = np.sqrt(np.sum((np.array([1.0, 2.0, 3.0]) - mean_y_true)**2))
    std_y_pred = np.sqrt(np.sum((np.array([1.1, 1.9, 3.2]) - mean_y_pred)**2))
    expected_pearson_corr = covariance / (std_y_true * std_y_pred)
    assert np.isclose(pearson_corr, expected_pearson_corr), f"Expected {expected_pearson_corr}, but got {pearson_corr}"

if __name__ == "__main__":
    pytest.main()    