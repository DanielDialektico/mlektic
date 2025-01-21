# Mlektic: A Simple and Efficient Machine Learning Library

Mlektic is a Python library built on top of TensorFlow, designed to simplify the implementation and experimentation with univariate/multivariate linear and logistic regression models. By providing a variety of gradient descent algorithms and regularization techniques, Mlektic enables both beginners and experienced practitioners to efficiently build, train, and evaluate regression models with minimal code.
<p align="center">
  <img src="https://raw.githubusercontent.com/DanielDialektico/mlektic/main/files/desc-1.jpg" alt="mlektic" width="200">
</p>

## Key Features

- **Linear and Logistic Regression**: Easily implement and experiment with these fundamental machine learning models.
- **Gradient Descent Variants**: Choose from various gradient descent methods, including batch, stochastic, and mini-batch, to optimize your models.
- **Regularization Techniques**: Apply L1, L2, or elastic net regularization to improve model generalization and prevent overfitting.
- **DataFrame Compatibility**: Seamlessly integrate with Pandas and Polars DataFrames, allowing you to preprocess and manage your data effortlessly.
- **Cost Visualization**: Visualize the evolution of the cost function over time using dynamic or static plots, helping you better understand the training process.
- **Evaluation Metrics**: Access a range of evaluation metrics to assess the performance of both linear and logistic regression models, ensuring that your models meet your performance criteria.
- **User-Friendly API**: Designed with simplicity in mind, Mlektic's API is intuitive and easy to use, making it accessible for users with varying levels of expertise.

## When to Use Mlektic?

- **Educational Purposes**: Ideal for students and educators to demonstrate the principles of regression and gradient descent in a practical setting.
- **Prototyping and Experimentation**: Quickly prototype regression models and experiment with different optimization techniques without the overhead of more complex machine learning frameworks.
- **Small to Medium Scale Projects**: Perfect for small to medium-sized projects where ease of use and quick iteration are more important than handling large-scale data.
<p align="center">
  <img src="https://raw.githubusercontent.com/DanielDialektico/mlektic/main/files/desc-2.jpg" alt="mlektic" width="200">
</p>

## Installation

You can install Mlektic using pip:

```sh
pip install mlektic
```  

## Getting Started
To train a model using linear regression with standard gradient descent and L1 regularization:

```python
    from mlektic.linear_reg import LinearRegressionArcht
    from mlektic import preprocessing
    from mlektic import methods
    import pandas as pd
    import numpy as np

    # Generate random data.
    np.random.seed(42)
    n_samples = 100
    feature1 = np.random.rand(n_samples)
    feature2 = np.random.rand(n_samples)
    target = 3 * feature1 + 5 * feature2 + np.random.randn(n_samples) * 0.5

    # Create pandas dataframe from the data.
    df = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'target': target
    })

    # Create train and test sets.
    train_set, test_set = preprocessing.pd_dataset(df, ['feature1', 'feature2'], 'target', 0.8)

    # Define regulizer and optimizer.
    regularizer = methods.regularizer_archt('l1', lambda_value=0.01)
    optimizer = methods.optimizer_archt('sgd-standard')

    # Configure the model.
    lin_reg = LinearRegressionArcht(iterations=50, optimizer=optimizer, regularizer=regularizer)

    # Train the model.
    lin_reg.train(train_set)
```
```plaintext
    Epoch 5, Loss: 15.191523551940918
    Epoch 10, Loss: 11.642797470092773
    Epoch 15, Loss: 9.021803855895996
    Epoch 20, Loss: 7.08500862121582
    Epoch 25, Loss: 5.652813911437988
    Epoch 30, Loss: 4.592779636383057
    Epoch 35, Loss: 3.807236909866333
    Epoch 40, Loss: 3.2241621017456055
    Epoch 45, Loss: 2.790440320968628
    Epoch 50, Loss: 2.4669017791748047
```

The cost evolution can be plotted with:
```sh
    from mlektic.plot_utils import plot_cost

    cost_history = lin_reg.get_cost_history()
    plot_cost(cost_history, dim = (7, 5))
```

<p>
  <img src="https://raw.githubusercontent.com/DanielDialektico/mlektic/main/files/plot.jpg" alt="cost plot" width="500">
</p>
<br><br/>

You can replace `LinearRegressionArcht` with `LogisticRegressionArcht`, and try different types of optimizers and regularizers.

## Documentation
For more detailed information, including API references and advanced usage, please refer to the [full documentation](https://dialektico.com/mlektic/docs/).

## Contributing
Contributions are welcome! If you have suggestions for improvements, feel free to open an issue or send me an email to contacto@dialektico.com.

## License
Mlektic is licensed under the Apache 2.0 License. See the [LICENSE](https://github.com/DanielDialektico/mlektic/blob/main/LICENSE) file for more details. 