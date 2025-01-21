import tensorflow as tf

def calculate_binary_crossentropy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculates the binary cross-entropy loss between true labels and predicted probabilities.

    Args:
        y_true (tf.Tensor): True labels. Shape should be (n_samples,).
        y_pred (tf.Tensor): Predicted probabilities. Shape should be (n_samples, 2).

    Returns:
        tf.Tensor: Binary cross-entropy loss.
    """
    y_pred_positive = y_pred[:, 1]
    return tf.keras.losses.binary_crossentropy(y_true, y_pred_positive)

def calculate_categorical_crossentropy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculates the categorical cross-entropy loss between true labels and predicted probabilities.

    Args:
        y_true (tf.Tensor): True labels, one-hot encoded. Shape should be (n_samples, num_classes).
        y_pred (tf.Tensor): Predicted probabilities. Shape should be (n_samples, num_classes).

    Returns:
        tf.Tensor: Categorical cross-entropy loss.
    """
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

def calculate_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculates the accuracy between true labels and predicted probabilities.

    Args:
        y_true (tf.Tensor): True labels, one-hot encoded. Shape should be (n_samples, num_classes).
        y_pred (tf.Tensor): Predicted probabilities. Shape should be (n_samples, num_classes).

    Returns:
        tf.Tensor: Accuracy.
    """
    predictions = tf.argmax(y_pred, axis=1)
    true_labels = tf.argmax(y_true, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, true_labels), dtype=tf.float32))

def calculate_precision(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculates the precision between true labels and predicted probabilities.

    Args:
        y_true (tf.Tensor): True labels, one-hot encoded. Shape should be (n_samples, num_classes).
        y_pred (tf.Tensor): Predicted probabilities. Shape should be (n_samples, num_classes).

    Returns:
        tf.Tensor: Precision.
    """
    predictions = tf.argmax(y_pred, axis=1)
    true_labels = tf.argmax(y_true, axis=1)
    true_positives = tf.reduce_sum(tf.cast(predictions * true_labels, dtype=tf.float32), axis=0)
    predicted_positives = tf.reduce_sum(tf.cast(predictions, dtype=tf.float32), axis=0)
    return true_positives / (predicted_positives + tf.keras.backend.epsilon())

def calculate_recall(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculates the recall between true labels and predicted probabilities.

    Args:
        y_true (tf.Tensor): True labels, one-hot encoded. Shape should be (n_samples, num_classes).
        y_pred (tf.Tensor): Predicted probabilities. Shape should be (n_samples, num_classes).

    Returns:
        tf.Tensor: Recall.
    """
    predictions = tf.argmax(y_pred, axis=1)
    true_labels = tf.argmax(y_true, axis=1)
    true_positives = tf.reduce_sum(tf.cast(predictions * true_labels, dtype=tf.float32), axis=0)
    possible_positives = tf.reduce_sum(tf.cast(true_labels, dtype=tf.float32), axis=0)
    return true_positives / (possible_positives + tf.keras.backend.epsilon())

def calculate_f1_score(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculates the F1 score between true labels and predicted probabilities.

    Args:
        y_true (tf.Tensor): True labels, one-hot encoded. Shape should be (n_samples, num_classes).
        y_pred (tf.Tensor): Predicted probabilities. Shape should be (n_samples, num_classes).

    Returns:
        tf.Tensor: F1 score.
    """
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

def calculate_confusion_matrix(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculates the confusion matrix between true labels and predicted probabilities.

    Args:
        y_true (tf.Tensor): True labels, one-hot encoded. Shape should be (n_samples, num_classes).
        y_pred (tf.Tensor): Predicted probabilities. Shape should be (n_samples, num_classes).

    Returns:
        tf.Tensor: Confusion matrix of shape (2, 2).
    """
    predictions = tf.argmax(y_pred, axis=1)
    true_labels = tf.argmax(y_true, axis=1)
    true_positives = tf.reduce_sum(tf.cast(predictions * true_labels, dtype=tf.float32), axis=0)
    false_positives = tf.reduce_sum(tf.cast(predictions * (1 - true_labels), dtype=tf.float32), axis=0)
    true_negatives = tf.reduce_sum(tf.cast((1 - predictions) * (1 - true_labels), dtype=tf.float32), axis=0)
    false_negatives = tf.reduce_sum(tf.cast((1 - predictions) * true_labels, dtype=tf.float32), axis=0)
    return tf.convert_to_tensor([[true_positives, false_positives], [false_negatives, true_negatives]], dtype=tf.float32)