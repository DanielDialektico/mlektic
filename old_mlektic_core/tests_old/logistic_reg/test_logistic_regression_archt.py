import pytest
import numpy as np
import tensorflow as tf
from mlektic.logistic_reg.logistic_regression_archt import LogisticRegressionArcht
from mlektic.methods import optimizer_archt

def test_initialization():
    model = LogisticRegressionArcht(iterations=1000, use_intercept=True, verbose=True)
    assert model.iterations == 1000
    assert model.use_intercept is True
    assert model.verbose is True
    assert model.weights is None
    assert model.cost_history is None
    assert model.metric_history is None
    assert model.n_features is None
    assert model.regularizer is None
    assert model.method == 'logistic'
    assert model.metric == 'accuracy'

def test_train_logistic():
    np.random.seed(42)
    x_train = np.random.rand(100, 2)
    y_train = np.random.randint(0, 2, 100)
    train_set = (x_train, y_train)

    model = LogisticRegressionArcht(iterations=50, method='logistic')
    model.train(train_set)
    
    assert model.weights is not None
    assert model.cost_history is None

def test_train_batch():
    np.random.seed(42)
    x_train = np.random.rand(100, 2)
    y_train = np.random.randint(0, 2, 100)
    train_set = (x_train, y_train)

    optimizer = optimizer_archt('sgd-standard', learning_rate=0.01)
    model = LogisticRegressionArcht(iterations=50, method='batch', optimizer=optimizer)
    model.train(train_set)
    
    assert model.weights is not None
    assert len(model.get_cost_history()) == 50
    assert len(model.get_metric_history()) == 50

def test_save_and_load_model(tmp_path):
    np.random.seed(42)
    x_train = np.random.rand(100, 2)
    y_train = np.random.randint(0, 2, 100)
    train_set = (x_train, y_train)

    optimizer = optimizer_archt('sgd-standard', learning_rate=0.01)
    model = LogisticRegressionArcht(iterations=50, method='batch', optimizer=optimizer)
    model.train(train_set)

    # Save model
    save_path = tmp_path / "logistic_regression_model.json"
    model.save_model(save_path)

    # Load model
    loaded_model = LogisticRegressionArcht()
    loaded_model.load_model(save_path)

    # Check if the loaded model has the same parameters
    np.testing.assert_array_almost_equal(model.get_parameters(), loaded_model.get_parameters())
    np.testing.assert_array_almost_equal(model.get_intercept(), loaded_model.get_intercept())
    assert model.n_features == loaded_model.n_features
    assert model.use_intercept == loaded_model.use_intercept
    assert model.num_classes == loaded_model.num_classes

def test_predict():
    np.random.seed(42)
    x_train = np.random.rand(100, 2)
    y_train = np.random.randint(0, 2, 100)
    train_set = (x_train, y_train)

    optimizer = optimizer_archt('sgd-standard', learning_rate=0.01)
    model = LogisticRegressionArcht(iterations=50, method='batch', optimizer=optimizer)
    model.train(train_set)

    x_new = np.array([0.5, 0.5])
    prediction = model.predict_prob(x_new)
    assert prediction.shape == (1, 2)  # As it's a softmax output for binary classification

def test_eval():
    np.random.seed(42)
    x_train = np.random.rand(100, 2)
    y_train = np.random.randint(0, 2, 100)
    train_set = (x_train, y_train)

    optimizer = optimizer_archt('sgd-standard', learning_rate=0.01)
    model = LogisticRegressionArcht(iterations=50, method='batch', optimizer=optimizer)
    model.train(train_set)

    accuracy = model.eval(train_set, 'accuracy')
    assert accuracy > 0

if __name__ == "__main__":
    pytest.main()    