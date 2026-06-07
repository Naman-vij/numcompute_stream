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
        assert np.abs(np.std(X_scaled[:, 0]) - 1.0) < 1e-6
    
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

class TestMinMaxScaler:
    
    def test_range_0_1(self):
        """All values should be in [0, 1]"""
        scaler = MinMaxScaler()
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        scaler.fit(X)
        
        X_scaled = scaler.transform(X)
        assert np.min(X_scaled) >= 0
        assert np.max(X_scaled) <= 1
    
    def test_partial_fit(self):
        """Min/max should update across chunks"""
        scaler = MinMaxScaler()
        X1 = np.array([[1, 2], [3, 4]], dtype=float)
        X2 = np.array([[5, 6], [7, 8]], dtype=float)
        
        scaler.partial_fit(X1)
        scaler.partial_fit(X2)
        
        assert scaler.min[0] == 1.0
        assert scaler.max[0] == 7.0
    
    def test_fit_transform(self):
        """fit_transform should work"""
        scaler = MinMaxScaler()
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        
        X_scaled = scaler.fit_transform(X)
        assert np.min(X_scaled) >= 0
        assert np.max(X_scaled) <= 1


class TestSimpleImputer:
    
    def test_mean_imputation(self):
        """Should replace NaN with mean"""
        imputer = SimpleImputer(strategy='mean')
        X = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        imputer.fit(X)
        
        X_imputed = imputer.transform(X)
        assert not np.any(np.isnan(X_imputed))
        assert X_imputed[1, 0] == 3.0
    
    def test_median_imputation(self):
        """Should replace NaN with median"""
        imputer = SimpleImputer(strategy='median')
        X = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        imputer.fit(X)
        
        X_imputed = imputer.transform(X)
        assert not np.any(np.isnan(X_imputed))
    
    def test_partial_fit(self):
        """Imputer should update with new chunks"""
        imputer = SimpleImputer(strategy='mean')
        X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        X2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        
        imputer.partial_fit(X1)
        imputer.partial_fit(X2)
        
        X_test = np.array([[np.nan, 2.0]])
        X_imputed = imputer.transform(X_test)
        assert not np.isnan(X_imputed[0, 0])


class TestOneHotEncoder:
    
    def test_binary_feature(self):
        """Two categories → 2 binary columns"""
        encoder = OneHotEncoder()
        X = np.array([['red'], ['blue'], ['red']])
        encoder.fit(X)
        
        X_encoded = encoder.transform(X)
        assert X_encoded.shape == (3, 2)
        assert X_encoded[0, 1] == 1  # First row is 'red'
        assert X_encoded[1, 0] == 1  # Second row is 'blue'
    
    def test_multiclass_feature(self):
        """Three categories → 3 binary columns"""
        encoder = OneHotEncoder()
        X = np.array([['red'], ['blue'], ['green']])
        encoder.fit(X)
        
        X_encoded = encoder.transform(X)
        assert X_encoded.shape == (3, 3)
    
    def test_partial_fit_new_category(self):
        """Should learn new categories in streaming"""
        encoder = OneHotEncoder()
        X1 = np.array([['red'], ['blue']])
        X2 = np.array([['green']])
        
        encoder.fit(X1)
        encoder.partial_fit(X2)
        
        assert len(encoder.categories[0]) == 3
    
    def test_fit_transform(self):
        """fit_transform should work"""
        encoder = OneHotEncoder()
        X = np.array([['a'], ['b'], ['a']])
        
        X_encoded = encoder.fit_transform(X)
        assert X_encoded.shape == (3, 2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])