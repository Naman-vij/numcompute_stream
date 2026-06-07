"""Unit tests for preprocessing.py"""

import numpy as np
import pytest
from numcompute_stream.preprocessing import (
    StandardScaler, MinMaxScaler, SimpleImputer, OneHotEncoder
)


class TestStandardScaler:
    
    def test_fit_means_zero(self):
        """After scaling, mean should be ~0"""
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        scaler.fit(X)
        
        X_scaled = scaler.transform(X)
        assert np.abs(np.mean(X_scaled[:, 0])) < 1e-10
        assert np.abs(np.mean(X_scaled[:, 1])) < 1e-10
    
    def test_fit_std_one(self):
        """After scaling, std should be ~1"""
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        scaler.fit(X)
        
        X_scaled = scaler.transform(X)
        assert np.abs(np.std(X_scaled[:, 0]) - 1.0) < 1e-10
    
    def test_partial_fit_streaming(self):
        """Scaler should work with streaming updates"""
        scaler = StandardScaler()
        X1 = np.array([[1, 2], [3, 4]], dtype=float)
        X2 = np.array([[5, 6], [7, 8]], dtype=float)
        
        scaler.partial_fit(X1)
        scaler.partial_fit(X2)
        
        X_scaled = scaler.transform(np.vstack([X1, X2]))
        assert X_scaled.shape == (4, 2)
    
    def test_with_nan(self):
        """Scaler should handle NaN values"""
        scaler = StandardScaler()
        X = np.array([[1, 2], [np.nan, 4], [5, 6]], dtype=float)
        scaler.fit(X)
        
        X_scaled = scaler.transform(X)
        assert X_scaled.shape == (3, 2)