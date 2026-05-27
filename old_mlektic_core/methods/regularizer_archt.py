import tensorflow as tf
from mlektic.methods.base.regularizers import Regularizers
from typing import Callable

def regularizer_archt(method: str = 'l1', lambda_value: float = 0.1, alpha: float = 0.5) -> Callable[[tf.Tensor], tf.Tensor]:
    """
    Returns a regularization function based on the specified method.

    Args:
        method (str, optional): The regularization method to use. Default is 'l1'.
                                Supported methods are 'l1', 'l2', 'elastic_net'.
        lambda_value (float, optional): Regularization parameter. Default is 0.1.
        alpha (float, optional): Mixing parameter for elastic net regularization, with 0 <= alpha <= 1. Default is 0.5.

    Returns:
        Callable[[tf.Tensor], tf.Tensor]: A function that takes weights as input and returns the regularization term.

    Raises:
        ValueError: If the specified method is not supported.
    """
    if method == 'l1':
        return Regularizers.l1(lambda_value)
    elif method == 'l2':
        return Regularizers.l2(lambda_value)
    elif method == 'elastic_net':
        return Regularizers.elastic_net(lambda_value, alpha)
    else:
        raise ValueError(f'{method} is not a valid value for regularizer.')