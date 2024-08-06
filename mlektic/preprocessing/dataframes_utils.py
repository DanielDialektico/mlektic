import pandas as pd
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, List

def pd_dataset(
    df: pd.DataFrame, 
    input_columns: List[str], 
    output_column: str, 
    train_fraction: float, 
    shuffle: bool = True, 
    random_seed: int = 42,
    normalize: bool = False,
    normalization_type: str = 'standard'
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Prepares train and test datasets from a pandas DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        input_columns (List[str]): List of column names to be used as inputs.
        output_column (str): Column name to be used as the output/target.
        train_fraction (float): Fraction of data to be used for training.
        shuffle (bool, optional): Whether to shuffle the data before splitting. Default is True.
        random_seed (int, optional): Seed for random number generator. Default is 42.
        normalize (bool, optional): Whether to normalize the input data. Default is False.
        normalization_type (str, optional): Type of normalization ('standard' or 'minmax'). Default is 'standard'.

    Returns:
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: Tuple of train and test datasets.

    Raises:
        ValueError: If `normalization_type` is not supported.
    """
    X = df[input_columns].values
    y = df[output_column].values

    if normalize:
        if normalization_type == 'standard':
            scaler = StandardScaler()
        elif normalization_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Unsupported normalization_type. Choose 'standard' or 'minmax'.")
        X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_fraction, random_state=random_seed, shuffle=shuffle)
    return (X_train, y_train), (X_test, y_test)

def pl_dataset(
    df: pl.DataFrame, 
    input_columns: List[str], 
    output_column: str, 
    train_fraction: float, 
    shuffle: bool = True, 
    random_seed: int = 42,
    normalize: bool = False,
    normalization_type: str = 'standard'
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Prepares train and test datasets from a polars DataFrame.

    Args:
        df (pl.DataFrame): DataFrame containing the data.
        input_columns (List[str]): List of column names to be used as inputs.
        output_column (str): Column name to be used as the output/target.
        train_fraction (float): Fraction of data to be used for training.
        shuffle (bool, optional): Whether to shuffle the data before splitting. Default is True.
        random_seed (int, optional): Seed for random number generator. Default is 42.
        normalize (bool, optional): Whether to normalize the input data. Default is False.
        normalization_type (str, optional): Type of normalization ('standard' or 'minmax'). Default is 'standard'.

    Returns:
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: Tuple of train and test datasets.

    Raises:
        ValueError: If `normalization_type` is not supported.
    """
    X = df.select(input_columns).to_numpy()
    y = df.select(output_column).to_numpy().flatten()

    if normalize:
        if normalization_type == 'standard':
            scaler = StandardScaler()
        elif normalization_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Unsupported normalization_type. Choose 'standard' or 'minmax'.")
        X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_fraction, random_state=random_seed, shuffle=shuffle)
    return (X_train, y_train), (X_test, y_test)