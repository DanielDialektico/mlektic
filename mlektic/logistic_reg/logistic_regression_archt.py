import tensorflow as tf
import numpy as np
import json
from typing import Union, Tuple, Callable
from .logreg_utils import calculate_categorical_crossentropy, calculate_accuracy, calculate_precision, calculate_recall, calculate_f1_score, calculate_confusion_matrix, calculate_binary_crossentropy

class LogisticRegressionArcht:
    """
    Logistic Regression model class supporting different training methods including Maximum Likelihood Estimation (MLE),
    batch gradient descent, stochastic gradient descent, and mini-batch gradient descent.

    Attributes:
        iterations (int): Number of training iterations (only applicable for gradient descent methods).
        use_intercept (bool): Whether to include an intercept in the model.
        verbose (bool): Whether to print training progress.
        weights (tf.Variable): Model weights.
        cost_history (list): History of cost values during training (only applicable for gradient descent methods).
        metric_history (list): History of metric values during training (only applicable for gradient descent methods).
        n_features (int): Number of features in the input data.
        regularizer (callable): Regularization function.
        optimizer (tf.optimizers.Optimizer): Optimizer for gradient descent.
        method (str): Training method to use.
        metric (str): Evaluation metric to use.
        num_classes (int): Number of classes in the target variable.

    Methods:
        train(train_set): Trains the model on the provided training set.
        get_parameters(): Returns the model parameters (weights).
        get_intercept(): Returns the model intercept.
        get_cost_history(): Returns the history of cost values.
        get_metric_history(): Returns the history of metric values.
        predict(input_new): Predicts output for new input data.
        eval(test_set, metric): Evaluates the model on a test set using the specified metric.
        save_model(filepath): Saves the model to a file.
        load_model(filepath): Loads the model from a file.
    """

    def __init__(self, iterations: int = 50, use_intercept: bool = True, verbose: bool = True, 
                 regularizer: Callable = None, optimizer: Union[Tuple[tf.optimizers.Optimizer, str, int], None] = None, method: str = 'mle', metric: str = 'accuracy'):
        """
        Initializes the LogisticRegressionArcht instance.

        Args:
            iterations (int, optional): Number of training iterations (only applicable for gradient descent methods). Default is 50.
            use_intercept (bool, optional): Whether to include an intercept in the model. Default is True.
            verbose (bool, optional): Whether to print training progress. Default is True.
            regularizer (callable, optional): Regularization function. Default is None.
            optimizer (tuple, optional): Tuple containing optimizer instance, method, and batch size. Default is None.
            method (str, optional): Training method to use. Options are 'mle', 'batch', 'stochastic', 'mini-batch'. Default is 'mle'.
            metric (str, optional): Evaluation metric to use. Options are 'accuracy', 'precision', 'recall', 'f1_score'. Default is 'accuracy'.
        
        Raises:
            ValueError: If the specified metric is not supported.
        """
        self.iterations = iterations
        self.use_intercept = use_intercept
        self.verbose = verbose
        self.weights = None
        self.cost_history = []
        self.metric_history = []
        self.n_features = None
        self.regularizer = regularizer
        self.method = method
        self.metric = metric
        self.num_classes = None

        if optimizer:
            self.optimizer, self.method, self.batch_size = optimizer
        else:
            self.optimizer = tf.optimizers.SGD(learning_rate=0.01)
            self.method = method if method in ['mle', 'batch', 'stochastic', 'mini-batch'] else 'mle'
            if self.method == 'mini-batch':
                self.batch_size = 32

        valid_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        if self.metric not in valid_metrics:
            raise ValueError(f"Unsupported metric '{self.metric}'. Supported metrics are: {valid_metrics}")

    def _softmax(self, z: tf.Tensor) -> tf.Tensor:
        """
        Applies the softmax function to the input tensor.

        Args:
            z (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Softmax-transformed tensor.
        """
        return tf.nn.softmax(z)

    def _predict(self, input: tf.Tensor) -> tf.Tensor:
        """
        Predicts the output probabilities using the logistic model.

        Args:
            input (tf.Tensor): Input tensor of shape (n_samples, n_features).

        Returns:
            tf.Tensor: Predicted probabilities tensor of shape (n_samples, num_classes).
        """
        logits = tf.matmul(input, self.weights)
        return self._softmax(logits)

    def _cost_function(self, input: tf.Tensor, output: tf.Tensor) -> tf.Tensor:
        """
        Computes the cost function (Categorical Cross-Entropy + Regularization).

        Args:
            input (tf.Tensor): Input tensor of shape (n_samples, n_features).
            output (tf.Tensor): True output tensor of shape (n_samples, num_classes).

        Returns:
            tf.Tensor: Cost value.
        """
        predictions = self._predict(input)
        categorical_crossentropy_loss = tf.reduce_mean(calculate_categorical_crossentropy(output, predictions))
        
        if self.regularizer:
            regularizer_loss = self.regularizer(self.weights)
        else:
            regularizer_loss = 0.0
        
        return categorical_crossentropy_loss + regularizer_loss

    def _compute_metric(self, input: tf.Tensor, output: tf.Tensor) -> tf.Tensor:
        """
        Computes the specified evaluation metric.

        Args:
            input (tf.Tensor): Input tensor of shape (n_samples, n_features).
            output (tf.Tensor): True output tensor of shape (n_samples, num_classes).

        Returns:
            tf.Tensor: Metric value.
        """
        predictions = self._predict(input)
        metrics = {
            'accuracy': calculate_accuracy,
            'precision': calculate_precision,
            'recall': calculate_recall,
            'f1_score': calculate_f1_score
        }
        
        return metrics[self.metric](output, predictions)

    def _train_mle(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Trains the model using the Maximum Likelihood Estimation (MLE) method.

        Args:
            x_train (np.ndarray): Training input data of shape (n_samples, n_features).
            y_train (np.ndarray): Training output data of shape (n_samples,).
        """
        x_train_t = tf.transpose(x_train)
        x_train_t_x_train = tf.matmul(x_train_t, x_train)
        x_train_t_y_train = tf.matmul(x_train_t, y_train)
        self.weights = tf.linalg.solve(x_train_t_x_train, x_train_t_y_train)
        self.weights = tf.cast(self.weights, tf.float32)
        if self.verbose:
            print("Model trained using Maximum Likelihood Estimation (MLE).")

    def _train_batch(self, x_train: tf.Tensor, y_train: tf.Tensor) -> None:
        """
        Trains the model using batch gradient descent.

        Args:
            x_train (tf.Tensor): Training input data of shape (n_samples, n_features).
            y_train (tf.Tensor): Training output data of shape (n_samples, num_classes).
        """
        for i in range(self.iterations):
            with tf.GradientTape() as tape:
                cost = self._cost_function(x_train, y_train)
                metric_value = self._compute_metric(x_train, y_train)
            gradients = tape.gradient(cost, [self.weights])
            self.optimizer.apply_gradients(zip(gradients, [self.weights]))
            
            # If cost or metric_value is an array, reduce it to a single value (mean)
            cost_value = np.mean(cost.numpy())
            metric_value_scalar = np.mean(metric_value.numpy())
            
            self.cost_history.append(cost_value)
            self.metric_history.append(metric_value_scalar)

            if self.verbose and (i + 1) % (self.iterations // 10) == 0:
                print(f'Epoch {i + 1}, Loss: {cost_value}, {self.metric.capitalize()}: {metric_value_scalar}')
        if self.verbose:
            print('\n')

    def _train_stochastic(self, x_train: tf.Tensor, y_train: tf.Tensor) -> None:
        """
        Trains the model using stochastic gradient descent.

        Args:
            x_train (tf.Tensor): Training input data of shape (n_samples, n_features).
            y_train (tf.Tensor): Training output data of shape (n_samples, num_classes).
        """
        for i in range(self.iterations):
            epoch_cost = 0
            epoch_metric = 0
            for j in range(x_train.shape[0]):
                with tf.GradientTape() as tape:
                    cost = self._cost_function(x_train[j:j+1], y_train[j:j+1])
                    metric_value = self._compute_metric(x_train[j:j+1], y_train[j:j+1])
                gradients = tape.gradient(cost, [self.weights])
                self.optimizer.apply_gradients(zip(gradients, [self.weights]))
                epoch_cost += cost.numpy()
                epoch_metric += metric_value.numpy()
            cost = epoch_cost / x_train.shape[0]
            metric_value = epoch_metric / x_train.shape[0]
            self.cost_history.append(cost)
            self.metric_history.append(metric_value)
            if self.verbose and (i + 1) % (self.iterations // 10) == 0:
                print(f'Epoch {i + 1}, Loss: {cost}, {self.metric.capitalize()}: {metric_value}')
        if self.verbose:
            print('\n')

    def _train_mini_batch(self, x_train: tf.Tensor, y_train: tf.Tensor) -> None:
        """
        Trains the model using mini-batch gradient descent.

        Args:
            x_train (tf.Tensor): Training input data of shape (n_samples, n_features).
            y_train (tf.Tensor): Training output data of shape (n_samples, num_classes).
        """
        for i in range(self.iterations):
            epoch_metric = 0
            for start in range(0, x_train.shape[0], self.batch_size):
                end = start + self.batch_size
                x_batch = x_train[start:end]
                y_batch = y_train[start:end]
                with tf.GradientTape() as tape:
                    cost = self._cost_function(x_batch, y_batch)
                    metric_value = self._compute_metric(x_batch, y_batch)
                gradients = tape.gradient(cost, [self.weights])
                self.optimizer.apply_gradients(zip(gradients, [self.weights]))
                epoch_metric += metric_value.numpy()
            self.cost_history.append(cost.numpy())
            self.metric_history.append(epoch_metric / (x_train.shape[0] // self.batch_size))
            if self.verbose and (i + 1) % (self.iterations // 10) == 0:
                print(f'Epoch {i + 1}, Loss: {cost.numpy()}, {self.metric.capitalize()}: {epoch_metric / (x_train.shape[0] // self.batch_size)}')
        if self.verbose:
            print('\n')

    def train(self, train_set: Tuple[np.ndarray, np.ndarray]) -> 'LogisticRegressionArcht':
        """
        Trains the model on the provided training set.

        Args:
            train_set (tuple): Tuple containing training input data (np.ndarray) and output data (np.ndarray).

        Returns:
            LogisticRegressionArcht: The trained model instance.
        """
        x_train, y_train = train_set
        x_train = x_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        
        self.num_classes = len(np.unique(y_train))
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=self.num_classes)
        
        if self.use_intercept:
            x_train = np.c_[np.ones((x_train.shape[0], 1)), x_train]
        self.n_features = x_train.shape[1]
        
        if self.method == 'mle':
            self._train_mle(x_train, y_train)
        else:
            self.weights = tf.Variable(tf.zeros([self.n_features, self.num_classes], dtype=tf.float32))
            x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
            y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
            if self.method == 'batch':
                self._train_batch(x_train, y_train)
            elif self.method == 'stochastic':
                self._train_stochastic(x_train, y_train)
            elif self.method == 'mini-batch':
                self._train_mini_batch(x_train, y_train)
            else:
                raise ValueError(f"Unsupported method '{self.method}'. Supported methods are 'mle', 'batch', 'stochastic', 'mini-batch'.")

    def get_parameters(self) -> np.ndarray:
        """
        Returns the model parameters (weights).

        Returns:
            np.ndarray: Array of model parameters.
        """
        if self.use_intercept:
            return self.weights[1:].numpy()
        else:
            return self.weights.numpy()

    def get_intercept(self) -> Union[np.ndarray, None]:
        """
        Returns the model intercept.

        Returns:
            Union[np.ndarray, None]: Intercept value if use_intercept is True, else None.
        """
        if self.use_intercept:
            return self.weights[0].numpy()
        else:
            return None

    def get_cost_history(self) -> list:
        """
        Returns the history of cost values during training.

        Returns:
            list: List of cost values.
        """
        return self.cost_history
    
    def get_metric_history(self) -> list:
        """
        Returns the history of metric values during training.

        Returns:
            list: List of metric values.
        """
        return self.metric_history
    
    def predict_prob(self, input_new: Union[np.ndarray, list, float]) -> np.ndarray:
        """
        Predicts output for new input data.

        Args:
            input_new (Union[np.ndarray, list, float]): New input data for prediction.

        Returns:
            np.ndarray: Predicted probabilities.
        
        Raises:
            ValueError: If the input does not have the expected number of features.
        """
        if self.use_intercept:
            if isinstance(input_new, float):
                input_new = np.array([[1] + [input_new] * (self.n_features - 1)], dtype=np.float32)
            elif isinstance(input_new, list):
                input_new = np.array([[1] + input_new], dtype=np.float32)
            elif isinstance(input_new, np.ndarray) and (input_new.ndim == 1):
                input_new = np.c_[np.ones((1, 1), dtype=np.float32), input_new.reshape(1, -1)]
            else:
                input_new = np.c_[np.ones((input_new.shape[0], 1), dtype=np.float32), input_new]
        else:
            if isinstance(input_new, float):
                input_new = np.array([[input_new] * self.n_features], dtype=np.float32)
            elif isinstance(input_new, list):
                input_new = np.array([input_new], dtype=np.float32)
            elif isinstance(input_new, np.ndarray) and (input_new.ndim == 1):
                input_new = input_new.reshape(1, -1).astype(np.float32)
        
        if input_new.shape[1] != self.n_features:
            raise ValueError(f"Expected input with {self.n_features} features, but got {input_new.shape[1]} features")
        
        input_new = tf.convert_to_tensor(input_new, dtype=tf.float32)
        return self._predict(input_new).numpy()

    def predict_class(self, input_new: Union[np.ndarray, list, float]) -> np.ndarray:
        """
        Predicts the class for new input data.

        Args:
            input_new (Union[np.ndarray, list, float]): New input data for prediction.

        Returns:
            np.ndarray: Predicted class(es).
        
        Raises:
            ValueError: If the input does not have the expected number of features.
        """
        if self.use_intercept:
            if isinstance(input_new, float):
                input_new = np.array([[1] + [input_new] * (self.n_features - 1)], dtype=np.float32)
            elif isinstance(input_new, list):
                input_new = np.array([[1] + input_new], dtype=np.float32)
            elif isinstance(input_new, np.ndarray) and (input_new.ndim == 1):
                input_new = np.c_[np.ones((1, 1), dtype=np.float32), input_new.reshape(1, -1)]
            else:
                input_new = np.c_[np.ones((input_new.shape[0], 1), dtype=np.float32), input_new]
        else:
            if isinstance(input_new, float):
                input_new = np.array([[input_new] * self.n_features], dtype=np.float32)
            elif isinstance(input_new, list):
                input_new = np.array([input_new], dtype=np.float32)
            elif isinstance(input_new, np.ndarray) and (input_new.ndim == 1):
                input_new = input_new.reshape(1, -1).astype(np.float32)

        if input_new.shape[1] != self.n_features:
            raise ValueError(f"Expected input with {self.n_features} features, but got {input_new.shape[1]} features")

        input_new = tf.convert_to_tensor(input_new, dtype=tf.float32)
        probabilities = self._predict(input_new).numpy()

        # Predice la clase con la mayor probabilidad
        predicted_classes = np.argmax(probabilities, axis=1)

        return predicted_classes


    def eval(self, test_set: Tuple[np.ndarray, np.ndarray], metric: str) -> float:
        """
        Evaluates the model on a test set using the specified metric.

        Args:
            test_set (tuple): Tuple containing test input data (np.ndarray) and output data (np.ndarray).
            metric (str): Metric to use for evaluation. Options are 'categorical_crossentropy', 'binary_crossentropy',
                        'accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix'.

        Returns:
            float: Evaluation result.
        
        Raises:
            ValueError: If the specified metric is not supported.
        """
        x_test, y_test = test_set
        x_test = x_test.astype(np.float32)
        y_test = y_test.astype(np.float32)
        
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=self.num_classes) if metric != 'binary_crossentropy' else y_test
        
        if self.use_intercept:
            x_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]
        
        x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
        y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
        
        y_pred = self._predict(x_test)
        
        metrics = {
            'categorical_crossentropy': calculate_categorical_crossentropy,
            'binary_crossentropy': calculate_binary_crossentropy,
            'accuracy': calculate_accuracy,
            'precision': calculate_precision,
            'recall': calculate_recall,
            'f1_score': calculate_f1_score,
            'confusion_matrix': calculate_confusion_matrix
        }

        if metric not in metrics:
            raise ValueError(f"Unsupported metric '{metric}'. Supported metrics are: {list(metrics.keys())}")
        
        metric_value = metrics[metric](y_test, y_pred)
        if metric != 'confusion_matrix':
            metric_value = tf.reduce_mean(metric_value)
        
        return metric_value.numpy()

    
    def save_model(self, filepath: str) -> None:
        """
        Saves the model to a file.

        Args:
            filepath (str): Path to the file where the model will be saved.
        """
        model_data = {
            'weights': self.weights.numpy().tolist(),
            'use_intercept': self.use_intercept,
            'n_features': self.n_features,
            'num_classes': self.num_classes,
            'cost_history': [float(cost) if isinstance(cost, (tf.Tensor, np.float32)) else cost for cost in self.cost_history],
            'metric_history': [float(metric) if isinstance(metric, (tf.Tensor, np.float32)) else metric for metric in self.metric_history]
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f)

    def load_model(self, filepath: str) -> None:
        """
        Loads the model from a file.

        Args:
            filepath (str): Path to the file from which the model will be loaded.
        """
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        self.weights = tf.Variable(np.array(model_data['weights'], dtype=np.float32))
        self.use_intercept = model_data['use_intercept']
        self.n_features = model_data['n_features']
        self.num_classes = model_data['num_classes']
        self.cost_history = [tf.convert_to_tensor(cost, dtype=tf.float32) if isinstance(cost, list) else tf.convert_to_tensor(cost, dtype=tf.float32) for cost in model_data['cost_history']]
        self.metric_history = [tf.convert_to_tensor(metric, dtype=tf.float32) if isinstance(metric, list) else tf.convert_to_tensor(metric, dtype=tf.float32) for metric in model_data['metric_history']]