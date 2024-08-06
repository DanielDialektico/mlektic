import pytest
import numpy as np
import tensorflow as tf
from mlektic.linear_reg.linear_regression_archt import LinearRegressionArcht
from mlektic.methods import optimizer_archt

def test_initialization():
    model = LinearRegressionArcht(iterations=1000, use_intercept=True, verbose=True)
    assert model.iterations == 1000
    assert model.use_intercept == True
    assert model.verbose == True
    assert model.weights is None
    assert model.cost_history is None
    assert model.n_features is None
    assert model.regularizer is None
    assert model.method == 'least_squares'

def test_train_least_squares():
    np.random.seed(42)
    x_train = np.random.rand(100, 2)
    y_train = 3 * x_train[:, 0] + 5 * x_train[:, 1] + np.random.randn(100) * 0.5
    train_set = (x_train, y_train)

    model = LinearRegressionArcht(iterations=50, method='least_squares')
    model.train(train_set)
    
    assert model.weights is not None
    assert model.cost_history is None

def test_train_batch():
    np.random.seed(42)
    x_train = np.random.rand(100, 2)
    y_train = 3 * x_train[:, 0] + 5 * x_train[:, 1] + np.random.randn(100) * 0.5
    train_set = (x_train, y_train)

    optimizer = optimizer_archt('sgd-standard', learning_rate=0.01)
    model = LinearRegressionArcht(iterations=50, optimizer=optimizer)
    model.train(train_set)
    
    assert model.weights is not None
    assert len(model.get_cost_history()) == 50

def test_save_and_load_model(tmp_path):
    np.random.seed(42)
    x_train = np.random.rand(100, 2)
    y_train = 3 * x_train[:, 0] + 5 * x_train[:, 1] + np.random.randn(100) * 0.5
    train_set = (x_train, y_train)

    optimizer = optimizer_archt('sgd-standard', learning_rate=0.01)
    model = LinearRegressionArcht(iterations=50, optimizer=optimizer)
    model.train(train_set)

    # Save model
    save_path = tmp_path / "linear_regression_model.json"
    model.save_model(save_path)

    # Load model
    loaded_model = LinearRegressionArcht()
    loaded_model.load_model(save_path)

    # Check if the loaded model has the same parameters
    np.testing.assert_array_almost_equal(model.get_parameters(), loaded_model.get_parameters())
    np.testing.assert_array_almost_equal(model.get_intercept(), loaded_model.get_intercept())
    assert model.n_features == loaded_model.n_features
    assert model.use_intercept == loaded_model.use_intercept

def test_predict():
    np.random.seed(42)
    x_train = np.random.rand(100, 2)
    y_train = 3 * x_train[:, 0] + 5 * x_train[:, 1] + np.random.randn(100) * 0.5
    train_set = (x_train, y_train)

    optimizer = optimizer_archt('sgd-standard', learning_rate=0.01)
    model = LinearRegressionArcht(iterations=50, optimizer=optimizer)
    model.train(train_set)

    x_new = np.array([0.5, 0.5])
    prediction = model.predict(x_new)
    assert prediction.shape == (1, 1)

def test_eval():
    np.random.seed(42)
    x_train = np.random.rand(100, 2)
    y_train = 3 * x_train[:, 0] + 5 * x_train[:, 1] + np.random.randn(100) * 0.5
    train_set = (x_train, y_train)

    optimizer = optimizer_archt('sgd-standard', learning_rate=0.01)
    model = LinearRegressionArcht(iterations=50, optimizer=optimizer)
    model.train(train_set)

    mse = model.eval(train_set, 'mse')
    assert mse > 0

if __name__ == "__main__":
    pytest.main()    