import pytest
import tensorflow as tf
from mlektic.methods.optimizer_archt import optimizer_archt

def test_sgd_standard():
    optimizer, method, batch_size = optimizer_archt(method='sgd-standard', learning_rate=0.01)
    assert isinstance(optimizer, tf.optimizers.SGD)
    assert optimizer.learning_rate.numpy() == pytest.approx(0.01)
    assert method == 'batch'
    assert batch_size is None

def test_sgd_stochastic():
    optimizer, method, batch_size = optimizer_archt(method='sgd-stochastic', learning_rate=0.01)
    assert isinstance(optimizer, tf.optimizers.SGD)
    assert optimizer.learning_rate.numpy() == pytest.approx(0.01)
    assert method == 'stochastic'
    assert batch_size is None

def test_sgd_mini_batch():
    optimizer, method, batch_size = optimizer_archt(method='sgd-mini-batch', learning_rate=0.01, batch_size=32)
    assert isinstance(optimizer, tf.optimizers.SGD)
    assert optimizer.learning_rate.numpy() == pytest.approx(0.01)
    assert method == 'mini-batch'
    assert batch_size == 32

def test_sgd_momentum():
    optimizer, method, batch_size = optimizer_archt(method='sgd-momentum', learning_rate=0.01, momentum=0.9)
    assert isinstance(optimizer, tf.optimizers.SGD)
    assert optimizer.learning_rate.numpy() == pytest.approx(0.01)
    assert optimizer.momentum == pytest.approx(0.9)
    assert method == 'batch'
    assert batch_size is None

def test_nesterov():
    optimizer, method, batch_size = optimizer_archt(method='nesterov', learning_rate=0.01, momentum=0.9, nesterov=True)
    assert isinstance(optimizer, tf.optimizers.SGD)
    assert optimizer.learning_rate.numpy() == pytest.approx(0.01)
    assert optimizer.momentum == pytest.approx(0.9)
    assert optimizer.nesterov
    assert method == 'batch'
    assert batch_size is None

def test_adagrad():
    optimizer, method, batch_size = optimizer_archt(method='adagrad', learning_rate=0.01)
    assert isinstance(optimizer, tf.optimizers.Adagrad)
    assert optimizer.learning_rate.numpy() == pytest.approx(0.01)
    assert method == 'batch'
    assert batch_size is None

def test_adadelta():
    optimizer, method, batch_size = optimizer_archt(method='adadelta', learning_rate=0.01)
    assert isinstance(optimizer, tf.optimizers.Adadelta)
    assert optimizer.learning_rate.numpy() == pytest.approx(0.01)
    assert method == 'batch'
    assert batch_size is None

def test_rmsprop():
    optimizer, method, batch_size = optimizer_archt(method='rmsprop', learning_rate=0.01)
    assert isinstance(optimizer, tf.optimizers.RMSprop)
    assert optimizer.learning_rate.numpy() == pytest.approx(0.01)
    assert method == 'batch'
    assert batch_size is None

def test_adam():
    optimizer, method, batch_size = optimizer_archt(method='adam', learning_rate=0.01)
    assert isinstance(optimizer, tf.optimizers.Adam)
    assert optimizer.learning_rate.numpy() == pytest.approx(0.01)
    assert method == 'batch'
    assert batch_size is None

def test_adamax():
    optimizer, method, batch_size = optimizer_archt(method='adamax', learning_rate=0.01)
    assert isinstance(optimizer, tf.optimizers.Adamax)
    assert optimizer.learning_rate.numpy() == pytest.approx(0.01)
    assert method == 'batch'
    assert batch_size is None

def test_nadam():
    optimizer, method, batch_size = optimizer_archt(method='nadam', learning_rate=0.01)
    assert isinstance(optimizer, tf.optimizers.Nadam)
    assert optimizer.learning_rate.numpy() == pytest.approx(0.01)
    assert method == 'batch'
    assert batch_size is None

def test_invalid_method():
    with pytest.raises(ValueError):
        optimizer_archt(method='invalid_method', learning_rate=0.01)
        
if __name__ == "__main__":
    pytest.main()