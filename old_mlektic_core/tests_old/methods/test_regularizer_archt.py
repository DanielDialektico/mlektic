import pytest
from mlektic.methods.base.regularizers import Regularizers
from mlektic.methods.regularizer_archt import regularizer_archt

def test_l1_regularizer():
    lambda_value = 0.1
    regularizer = regularizer_archt(method='l1', lambda_value=lambda_value)
    weights = [1.0, -1.0, 2.0, -2.0]
    regularization_value = regularizer(weights)
    expected_value = lambda_value * sum(abs(w) for w in weights)
    assert regularization_value.numpy() == pytest.approx(expected_value)

def test_l2_regularizer():
    lambda_value = 0.1
    regularizer = regularizer_archt(method='l2', lambda_value=lambda_value)
    weights = [1.0, -1.0, 2.0, -2.0]
    regularization_value = regularizer(weights)
    expected_value = lambda_value * sum(w ** 2 for w in weights)
    assert regularization_value.numpy() == pytest.approx(expected_value)

def test_elastic_net_regularizer():
    lambda_value = 0.1
    alpha = 0.5
    regularizer = regularizer_archt(method='elastic_net', lambda_value=lambda_value, alpha=alpha)
    weights = [1.0, -1.0, 2.0, -2.0]
    l1_value = lambda_value * alpha * sum(abs(w) for w in weights)
    l2_value = lambda_value * (1 - alpha) * sum(w ** 2 for w in weights)
    expected_value = l1_value + l2_value
    regularization_value = regularizer(weights)
    assert regularization_value.numpy() == pytest.approx(expected_value)

def test_invalid_method():
    with pytest.raises(ValueError):
        regularizer_archt(method='invalid_method')
        
if __name__ == "__main__":
    pytest.main()