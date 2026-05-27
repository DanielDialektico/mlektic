import tensorflow as tf

def calculate_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculates the Mean Squared Error (MSE) between the true and predicted values.

    Args:
        y_true (tf.Tensor): True values. Shape should be (n_samples, 1).
        y_pred (tf.Tensor): Predicted values. Shape should be (n_samples, 1).

    Returns:
        tf.Tensor: Mean Squared Error.
    """
    return tf.reduce_mean(tf.square(y_pred - y_true), axis=0)

def calculate_rmse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculates the Root Mean Squared Error (RMSE) between the true and predicted values.

    Args:
        y_true (tf.Tensor): True values. Shape should be (n_samples, 1).
        y_pred (tf.Tensor): Predicted values. Shape should be (n_samples, 1).

    Returns:
        tf.Tensor: Root Mean Squared Error.
    """
    mse = calculate_mse(y_true, y_pred)
    return tf.sqrt(mse)

def calculate_mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculates the Mean Absolute Error (MAE) between the true and predicted values.

    Args:
        y_true (tf.Tensor): True values. Shape should be (n_samples, 1).
        y_pred (tf.Tensor): Predicted values. Shape should be (n_samples, 1).

    Returns:
        tf.Tensor: Mean Absolute Error.
    """
    return tf.reduce_mean(tf.abs(y_pred - y_true), axis=0)

def calculate_mape(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculates the Mean Absolute Percentage Error (MAPE) between the true and predicted values.

    Args:
        y_true (tf.Tensor): True values. Shape should be (n_samples, 1).
        y_pred (tf.Tensor): Predicted values. Shape should be (n_samples, 1).

    Returns:
        tf.Tensor: Mean Absolute Percentage Error.
    
    Raises:
        ValueError: If any value in y_true is zero, which would result in division by zero.
    """
    if tf.reduce_any(y_true == 0):
        raise ValueError("y_true contains zero values, which would result in division by zero.")
    return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true), axis=0) * 100

def calculate_r2(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculates the R-squared (R2) score, which indicates the proportion of the variance in the dependent variable
    that is predictable from the independent variable(s).

    Args:
        y_true (tf.Tensor): True values. Shape should be (n_samples, 1).
        y_pred (tf.Tensor): Predicted values. Shape should be (n_samples, 1).

    Returns:
        tf.Tensor: R-squared score.
    """
    total_sum_of_squares = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true, axis=0)), axis=0)
    residual_sum_of_squares = tf.reduce_sum(tf.square(y_true - y_pred), axis=0)
    return 1 - (residual_sum_of_squares / total_sum_of_squares)

def calculate_pearson_correlation(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculates the Pearson correlation coefficient between the true and predicted values.

    Args:
        y_true (tf.Tensor): True values. Shape should be (n_samples, 1).
        y_pred (tf.Tensor): Predicted values. Shape should be (n_samples, 1).

    Returns:
        tf.Tensor: Pearson correlation coefficient.
    """
    mean_y_true = tf.reduce_mean(y_true, axis=0)
    mean_y_pred = tf.reduce_mean(y_pred, axis=0)
    covariance = tf.reduce_sum((y_true - mean_y_true) * (y_pred - mean_y_pred), axis=0)
    std_y_true = tf.sqrt(tf.reduce_sum(tf.square(y_true - mean_y_true), axis=0))
    std_y_pred = tf.sqrt(tf.reduce_sum(tf.square(y_pred - mean_y_pred), axis=0))
    return covariance / (std_y_true * std_y_pred)