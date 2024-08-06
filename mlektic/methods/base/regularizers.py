import tensorflow as tf

class Regularizers:
    """
    A class that provides static methods for various types of regularization.

    Methods:
    -------
    l1(lambda_value: float) -> Callable[[tf.Tensor], tf.Tensor]
        Returns a function that computes the L1 regularization term for a given tensor of weights.

    l2(lambda_value: float) -> Callable[[tf.Tensor], tf.Tensor]
        Returns a function that computes the L2 regularization term for a given tensor of weights.

    elastic_net(lambda_value: float, alpha: float) -> Callable[[tf.Tensor], tf.Tensor]
        Returns a function that computes the Elastic Net regularization term for a given tensor of weights.
    """

    @staticmethod
    def l1(lambda_value: float):
        """
        Returns a function that computes the L1 regularization term for a given tensor of weights.

        Args:
            lambda_value (float): The regularization parameter.

        Returns:
            Callable[[tf.Tensor], tf.Tensor]: A function that takes a tensor of weights as input and returns the L1 regularization term.
        """
        def regularization(weights: tf.Tensor) -> tf.Tensor:
            return lambda_value * tf.reduce_sum(tf.abs(weights), axis=0)
        return regularization

    @staticmethod
    def l2(lambda_value: float):
        """
        Returns a function that computes the L2 regularization term for a given tensor of weights.

        Args:
            lambda_value (float): The regularization parameter.

        Returns:
            Callable[[tf.Tensor], tf.Tensor]: A function that takes a tensor of weights as input and returns the L2 regularization term.
        """
        def regularization(weights: tf.Tensor) -> tf.Tensor:
            return lambda_value * tf.reduce_sum(tf.square(weights), axis=0)
        return regularization

    @staticmethod
    def elastic_net(lambda_value: float, alpha: float):
        """
        Returns a function that computes the Elastic Net regularization term for a given tensor of weights.

        Args:
            lambda_value (float): The regularization parameter.
            alpha (float): The mixing parameter between L1 and L2 regularization, with 0 <= alpha <= 1.

        Returns:
            Callable[[tf.Tensor], tf.Tensor]: A function that takes a tensor of weights as input and returns the Elastic Net regularization term.
        """
        def regularization(weights: tf.Tensor) -> tf.Tensor:
            l1_loss = lambda_value * alpha * tf.reduce_sum(tf.abs(weights), axis=0)
            l2_loss = lambda_value * (1 - alpha) * tf.reduce_sum(tf.square(weights), axis=0)
            return l1_loss + l2_loss
        return regularization