import tensorflow as tf
import numpy as np
import json
from .linreg_utils import calculate_mse, calculate_mae, calculate_mape, calculate_pearson_correlation, calculate_r2, calculate_rmse

class LinearRegressionArcht:
    """
    Linear Regression model class supporting different training methods including least squares, batch gradient descent,
    stochastic gradient descent, and mini-batch gradient descent.

    Attributes:
        iterations (int): Number of training iterations.
        use_intercept (bool): Whether to include an intercept in the model.
        verbose (bool): Whether to print training progress.
        weights (tf.Variable): Model weights.
        cost_history (list): History of cost values during training.
        n_features (int): Number of features in the input data.
        regularizer (callable): Regularization function.
        optimizer (tf.optimizers.Optimizer): Optimizer for gradient descent.
        method (str): Training method to use.

    Methods:
        train(train_set): Trains the model on the provided training set.
        get_parameters(): Returns the model parameters (weights).
        get_intercept(): Returns the model intercept.
        get_cost_history(): Returns the history of cost values.
        predict(input_new): Predicts output for new input data.
        eval(test_set, metric): Evaluates the model on a test set using the specified metric.
        save_model(filepath): Saves the model to a file.
        load_model(filepath): Loads the model from a file.
    """

    def __init__(self, iterations: int = 50, use_intercept: bool = True, verbose: bool = True, 
                 regularizer: callable = None, optimizer: tuple = None, method: str = 'least_squares'):
        """
        Initializes the LinearRegressionArcht instance.

        Args:
            iterations (int, optional): Number of training iterations. Default is 50.
            use_intercept (bool, optional): Whether to include an intercept in the model. Default is True.
            verbose (bool, optional): Whether to print training progress. Default is True.
            regularizer (callable, optional): Regularization function. Default is None.
            optimizer (tuple, optional): Tuple containing optimizer instance, method, and batch size. Default is None.
            method (str, optional): Training method to use. Options are 'least_squares', 'batch', 'stochastic', 'mini-batch'. Default is 'least_squares'.
        """
        self.iterations = iterations
        self.use_intercept = use_intercept
        self.verbose = verbose
        self.weights = None
        self.cost_history = None
        self.n_features = None
        self.regularizer = regularizer
        self.method = method

        if optimizer:
            self.optimizer, self.method, self.batch_size = optimizer
        else:
            self.optimizer = tf.optimizers.SGD(learning_rate=0.01)
            self.method = 'least_squares' if method == 'least_squares' else 'batch'
            if self.method != 'least_squares':
                self.batch_size = 32

    def _predict(self, input: tf.Tensor) -> tf.Tensor:
        """
        Predicts the output using the linear model.

        Args:
            input (tf.Tensor): Input tensor of shape (n_samples, n_features).

        Returns:
            tf.Tensor: Predicted output tensor of shape (n_samples, 1).
        """
        return tf.matmul(input, self.weights)

    def _cost_function(self, input: tf.Tensor, output: tf.Tensor) -> tf.Tensor:
        """
        Computes the cost function (Mean Squared Error + Regularization).

        Args:
            input (tf.Tensor): Input tensor of shape (n_samples, n_features).
            output (tf.Tensor): Output tensor of shape (n_samples, 1).

        Returns:
            tf.Tensor: Cost value.
        """
        predictions = self._predict(input)
        mse_loss = tf.reduce_mean(calculate_mse(output, predictions))
        
        if self.regularizer:
            regularizer_loss = self.regularizer(self.weights)
        else:
            regularizer_loss = 0.0
        
        return mse_loss + regularizer_loss

    def _train_least_squares(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Trains the model using the least squares method.

        Args:
            x_train (np.ndarray): Training input data of shape (n_samples, n_features).
            y_train (np.ndarray): Training output data of shape (n_samples,).
        """
        x_train_t = tf.transpose(x_train)
        x_train_t_x_train = tf.matmul(x_train_t, x_train)
        x_train_t_y_train = tf.matmul(x_train_t, y_train.reshape(-1, 1))
        self.weights = tf.linalg.solve(x_train_t_x_train, x_train_t_y_train)
        self.weights = tf.cast(self.weights, tf.float32)
        self.cost_history = None
        if self.verbose:
            print("Model trained using least squares method.")

    def _train_batch(self, x_train: tf.Tensor, y_train: tf.Tensor) -> None:
        """
        Trains the model using batch gradient descent.

        Args:
            x_train (tf.Tensor): Training input data of shape (n_samples, n_features).
            y_train (tf.Tensor): Training output data of shape (n_samples, 1).
        """
        for i in range(self.iterations):
            with tf.GradientTape() as tape:
                cost = self._cost_function(x_train, y_train)
            gradients = tape.gradient(cost, [self.weights])
            self.optimizer.apply_gradients(zip(gradients, [self.weights]))
            
            self.cost_history.append(cost.numpy().item())

            if self.verbose and (i + 1) % (self.iterations // 10) == 0:
                print(f'Epoch {i + 1}, Loss: {cost.numpy().item()}')

    def _train_stochastic(self, x_train: tf.Tensor, y_train: tf.Tensor) -> None:
        """
        Trains the model using stochastic gradient descent.

        Args:
            x_train (tf.Tensor): Training input data of shape (n_samples, n_features).
            y_train (tf.Tensor): Training output data of shape (n_samples, 1).
        """
        for i in range(self.iterations):
            epoch_cost = 0
            for j in range(x_train.shape[0]):
                with tf.GradientTape() as tape:
                    cost = self._cost_function(x_train[j:j+1], y_train[j:j+1])
                gradients = tape.gradient(cost, [self.weights])
                self.optimizer.apply_gradients(zip(gradients, [self.weights]))
                epoch_cost += cost.numpy()
            cost = epoch_cost / x_train.shape[0]
            self.cost_history.append(cost)
            if self.verbose and (i + 1) % (self.iterations // 10) == 0:
                print(f'Epoch {i + 1}, Loss: {cost}')
        if self.verbose:
            print('\n')

    def _train_mini_batch(self, x_train: tf.Tensor, y_train: tf.Tensor) -> None:
        """
        Trains the model using mini-batch gradient descent.

        Args:
            x_train (tf.Tensor): Training input data of shape (n_samples, n_features).
            y_train (tf.Tensor): Training output data of shape (n_samples, 1).
        """
        for i in range(self.iterations):
            for start in range(0, x_train.shape[0], self.batch_size):
                end = start + self.batch_size
                x_batch = x_train[start:end]
                y_batch = y_train[start:end]
                with tf.GradientTape() as tape:
                    cost = self._cost_function(x_batch, y_batch)
                gradients = tape.gradient(cost, [self.weights])
                self.optimizer.apply_gradients(zip(gradients, [self.weights]))
            self.cost_history.append(cost)
            if self.verbose and (i + 1) % (self.iterations // 10) == 0:
                print(f'Epoch {i + 1}, Loss: {cost}')
        if self.verbose:
            print('\n')

    def train(self, train_set: tuple) -> 'LinearRegressionArcht':
        """
        Trains the model on the provided training set.

        Args:
            train_set (tuple): Tuple containing training input data (np.ndarray) and output data (np.ndarray).

        Returns:
            LinearRegressionArcht: The trained model instance.
        """
        x_train, y_train = train_set
        x_train = x_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        if self.use_intercept:
            x_train = np.c_[np.ones((x_train.shape[0], 1)), x_train]
        self.n_features = x_train.shape[1]
        
        if self.method == 'least_squares':
            self._train_least_squares(x_train, y_train)
        else:
            self.weights = tf.Variable(tf.zeros([self.n_features, 1], dtype=tf.float32))
            self.cost_history = []
            x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
            y_train = tf.convert_to_tensor(y_train.reshape(-1, 1), dtype=tf.float32)
            if self.method == 'batch':
                self._train_batch(x_train, y_train)
            elif self.method == 'stochastic':
                self._train_stochastic(x_train, y_train)
            elif self.method == 'mini-batch':
                self._train_mini_batch(x_train, y_train)
            else:
                raise ValueError(f"Unsupported method '{self.method}'. Supported methods are 'batch', 'stochastic', 'mini-batch'.")

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

    def get_intercept(self) -> float:
        """
        Returns the model intercept.

        Returns:
            float: Intercept value if use_intercept is True, else None.
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
    
    def predict(self, input_new: 'Union[np.ndarray, list, float]') -> np.ndarray:
        """
        Predicts output for new input data.

        Args:
            input_new (Union[np.ndarray, list, float]): New input data for prediction.

        Returns:
            np.ndarray: Predicted output.
        
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

    def eval(self, test_set: tuple, metric: str) -> float:
        """
        Evaluates the model on a test set using the specified metric.

        Args:
            test_set (tuple): Tuple containing test input data (np.ndarray) and output data (np.ndarray).
            metric (str): Metric to use for evaluation. Options are 'mse', 'rmse', 'mae', 'mape', 'r2', 'corr'.

        Returns:
            float: Evaluation result.
        
        Raises:
            ValueError: If the specified metric is not supported.
        """
        x_test, y_test = test_set
        x_test = x_test.astype(np.float32)
        y_test = y_test.astype(np.float32)
        if self.use_intercept:
            x_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]
        
        x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
        y_test = tf.convert_to_tensor(y_test.reshape(-1, 1), dtype=tf.float32)
        
        y_pred = self._predict(x_test)
        
        metrics = {
            'mse': calculate_mse,
            'rmse': calculate_rmse,
            'mae': calculate_mae,
            'mape': calculate_mape,
            'r2': calculate_r2,
            'corr': calculate_pearson_correlation
        }

        if metric not in metrics:
            raise ValueError(f"Unsupported metric '{metric}'. Supported metrics are: {list(metrics.keys())}")
        
        return tf.reduce_mean(metrics[metric](y_test, y_pred)).numpy()
    
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
            'cost_history': [cost.numpy().tolist() if isinstance(cost, tf.Tensor) else cost for cost in self.cost_history]
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
        self.cost_history = [tf.convert_to_tensor(cost) if isinstance(cost, list) else cost for cost in model_data['cost_history']]