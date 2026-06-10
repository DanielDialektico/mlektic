import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from mlektic.preprocessing.dataframes_utils import pd_dataset

def test_pd_dataset():
    # Crear un DataFrame de ejemplo con pandas
    np.random.seed(42)
    n_samples = 100
    df = pd.DataFrame({
        'feature1': np.random.rand(n_samples),
        'feature2': np.random.rand(n_samples),
        'target': np.random.randint(0, 2, size=n_samples)
    })

    # Probar la normalización con MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(df[['feature1', 'feature2']])
    expected_data = scaler.transform(df[['feature1', 'feature2']])

    # Usar train_test_split sin shuffle para dividir los datos esperados de la misma manera
    X_train, X_test, _, _ = train_test_split(expected_data, df['target'].values, train_size=0.8, random_state=42, shuffle=False)

    # Llamar a la función pd_dataset
    train_set, test_set = pd_dataset(df, ['feature1', 'feature2'], 'target', 0.8, normalize=True, normalization_type='minmax', shuffle=False)
    
    # Verificar que los datos normalizados son consistentes
    np.testing.assert_allclose(train_set[0], X_train, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(test_set[0], X_test, rtol=1e-5, atol=1e-8)

if __name__ == "__main__":
    pytest.main()