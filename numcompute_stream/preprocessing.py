"""
preprocessing.py - Data Preprocessing with Streaming Support

Handles scaling, imputation, and encoding with streaming capability.
"""

import numpy as np
from typing import Optional, List


class StandardScaler:
    """
    Standardize features: (X - mean) / std
    
    Why: Makes all features comparable regardless of original scale.
    Example: Age (0-100) and Income (0-1000000) become comparable.
    
    Streaming: Updates mean/variance as new chunks arrive.
    """
    
    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        
        self.mean = None
        self.var = None
        self.n_features_in = None
        self.n_samples_seen = 0
        self.M2 = None
    
    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """Learn mean and variance from data (batch mode)."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_features_in = X.shape[1]
        self.mean = np.nanmean(X, axis=0)
        self.var = np.nanvar(X, axis=0)
        self.n_samples_seen = X.shape[0]
        
        return self
    
    def partial_fit(self, X: np.ndarray) -> 'StandardScaler':
        """Update mean and variance with new chunk (streaming mode)."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if self.n_features_in is None:
            self.n_features_in = X.shape[1]
            self.mean = np.zeros(self.n_features_in)
            self.M2 = np.zeros(self.n_features_in)
            self.n_samples_seen = 0
        
        # Welford's algorithm: update mean and variance without storing all data
        for i in range(X.shape[0]):
            for j in range(self.n_features_in):
                if not np.isnan(X[i, j]):
                    self.n_samples_seen += 1
                    delta = X[i, j] - self.mean[j]
                    self.mean[j] += delta / self.n_samples_seen
                    delta2 = X[i, j] - self.mean[j]
                    self.M2[j] += delta * delta2
        
        # Calculate variance from M2
        if self.n_samples_seen > 1:
            self.var = self.M2 / (self.n_samples_seen - 1)
        else:
            self.var = np.zeros(self.n_features_in)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale new data using learned mean and variance."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X_scaled = np.copy(X).astype(float)
        
        # Subtract mean
        if self.with_mean:
            X_scaled = X_scaled - self.mean
        
        # Divide by standard deviation
        if self.with_std:
            std = np.sqrt(np.maximum(self.var, 1e-10))
            X_scaled = X_scaled / std
        
        return X_scaled
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class MinMaxScaler:
    """
    Scale features to [0, 1] range: (X - min) / (max - min)
    
    Why: Some algorithms prefer bounded features in [0, 1].
    Example: [10, 50, 100] → [0, 0.44, 1.0]
    
    Streaming: Updates min/max as new chunks arrive.
    """
    
    def __init__(self):
        self.min = None
        self.max = None
        self.n_features_in = None
    
    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        """Learn min and max from data."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_features_in = X.shape[1]
        self.min = np.nanmin(X, axis=0)
        self.max = np.nanmax(X, axis=0)
        
        return self
    
    def partial_fit(self, X: np.ndarray) -> 'MinMaxScaler':
        """Update min and max with new chunk."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if self.n_features_in is None:
            self.n_features_in = X.shape[1]
            self.min = np.full(self.n_features_in, np.inf)
            self.max = np.full(self.n_features_in, -np.inf)
        
        # Update global min/max
        chunk_min = np.nanmin(X, axis=0)
        chunk_max = np.nanmax(X, axis=0)
        
        self.min = np.minimum(self.min, chunk_min)
        self.max = np.maximum(self.max, chunk_max)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale to [0, 1]."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X_scaled = np.copy(X).astype(float)
        range_ = np.maximum(self.max - self.min, 1e-10)
        X_scaled = (X_scaled - self.min) / range_
        
        return X_scaled
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X).transform(X)


class SimpleImputer:
    """
    Fill missing values (NaN) with mean/median/most_frequent.
    
    Why: Trees can't split on NaN. Must replace missing values.
    
    Strategies:
    - mean: Replace with average of column
    - median: Replace with middle value of column
    - most_frequent: Replace with most common value
    
    Example:
    Age: [25, NaN, 30, 35]
    Mean: 30
    Result: [25, 30, 30, 35]
    """
    
    def __init__(self, strategy: str = 'mean'):
        if strategy not in ['mean', 'median', 'most_frequent']:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        self.strategy = strategy
        self.fill_value = None
        self.n_features_in = None
    
    def fit(self, X: np.ndarray) -> 'SimpleImputer':
        """Learn fill values from data."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_features_in = X.shape[1]
        
        if self.strategy == 'mean':
            self.fill_value = np.nanmean(X, axis=0)
        elif self.strategy == 'median':
            self.fill_value = np.nanmedian(X, axis=0)
        elif self.strategy == 'most_frequent':
            self.fill_value = np.array([
                np.bincount(X[:, j].astype(int)[~np.isnan(X[:, j])]).argmax()
                if np.sum(~np.isnan(X[:, j])) > 0 else 0
                for j in range(self.n_features_in)
            ])
        
        return self
    
    def partial_fit(self, X: np.ndarray) -> 'SimpleImputer':
        """Update fill values with new chunk."""
        if self.fill_value is None:
            return self.fit(X)
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if self.strategy == 'mean':
            chunk_mean = np.nanmean(X, axis=0)
            self.fill_value = 0.5 * self.fill_value + 0.5 * chunk_mean
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Replace NaN with learned values."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X_imputed = np.copy(X)
        
        for j in range(self.n_features_in):
            mask = np.isnan(X[:, j])
            X_imputed[mask, j] = self.fill_value[j]
        
        return X_imputed
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X).transform(X)


class OneHotEncoder:
    """
    Convert categorical (text) features to binary columns.
    
    Why: Trees work with numbers, not text.
    
    Example:
    Original column: ['Red', 'Blue', 'Red']
    
    After encoding:
    Red_col:   [1, 0, 1]
    Blue_col:  [0, 1, 0]
    
    Streaming: Learns new categories as they appear in new chunks.
    """
    
    def __init__(self, sparse: bool = False):
        self.sparse = sparse
        self.categories = None
        self.n_features_in = None
    
    def fit(self, X: np.ndarray) -> 'OneHotEncoder':
        """Learn all unique categories from data."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_features_in = X.shape[1]
        # For each feature, find all unique values
        self.categories = [np.unique(X[:, j]) for j in range(self.n_features_in)]
        
        return self
    
    def partial_fit(self, X: np.ndarray) -> 'OneHotEncoder':
        """Update categories when new values appear."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if self.categories is None:
            return self.fit(X)
        
        # Add any new categories we haven't seen before
        for j in range(self.n_features_in):
            unique_vals = np.unique(X[:, j])
            self.categories[j] = np.unique(np.concatenate([self.categories[j], unique_vals]))
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Convert to binary columns."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = X.shape[0]
        n_output_features = sum(len(cats) for cats in self.categories)
        
        X_encoded = np.zeros((n_samples, n_output_features))
        
        # For each feature, create binary columns
        col_idx = 0
        for feature_idx in range(self.n_features_in):
            for cat_idx, category in enumerate(self.categories[feature_idx]):
                # Mark 1 where value matches category, 0 otherwise
                mask = X[:, feature_idx] == category
                X_encoded[mask, col_idx] = 1
                col_idx += 1
        
        return X_encoded
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X).transform(X)