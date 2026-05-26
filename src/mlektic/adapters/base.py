"""Adapter pattern for machine learning models."""

import abc
from typing import Any, Tuple, Dict, Optional
import numpy as np

class BaseModelAdapter(abc.ABC):
    """Abstract base class for all ML model adapters."""
    
    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values or class labels."""
        pass
        
    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray, classes: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict class probabilities."""
        pass
        
    @abc.abstractmethod
    def extract_linear_theta(self, d_expected: Optional[int] = None) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Extract linear regression weights and bias."""
        pass
        
    @abc.abstractmethod
    def extract_logistic_theta(self, d_expected: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Extract logistic regression weights and bias in normalized schema."""
        pass
        
    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model."""
        pass
        
    @abc.abstractmethod
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Incrementally fit the model if supported."""
        pass
        
    @property
    @abc.abstractmethod
    def is_iterative(self) -> bool:
        """Check if the model supports iterative training (partial_fit/warm_start)."""
        pass
        
    @property
    @abc.abstractmethod
    def classes(self) -> Optional[np.ndarray]:
        """Get the learned classes if classification."""
        pass
        
    @abc.abstractmethod
    def get_scaler_params(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns (mean, scale) used by the adapter's scaler, if any."""
        pass

    @abc.abstractmethod
    def transform_X(self, X: np.ndarray) -> np.ndarray:
        """Transform X through the pipeline up to the final estimator."""
        pass
