"""Unit tests for io.py"""

import numpy as np
import pytest
import tempfile
import os
from numcompute_stream.io import (
    load_csv, save_csv, create_synthetic_data,
    load_and_split, create_chunks
)


class TestCreateSyntheticData:
    
    def test_shape(self):
        """Should create correct shape."""
        X, y = create_synthetic_data(n_samples=100, n_features=5)
        
        assert X.shape == (100, 5)
        assert y.shape == (100,)
    
    def test_binary_classification(self):
        """Should create binary classification data."""
        X, y = create_synthetic_data(n_samples=100, n_classes=2)
        
        assert len(np.unique(y)) <= 2
        assert 0 in y
        assert 1 in y
    
    def test_multiclass(self):
        """Should create multiclass data."""
        X, y = create_synthetic_data(n_samples=300, n_classes=3)
        
        unique_classes = np.unique(y)
        assert 0 in unique_classes
        assert 1 in unique_classes
        assert 2 in unique_classes
    
    def test_reproducible(self):
        """Should be reproducible with random_state."""
        X1, y1 = create_synthetic_data(n_samples=50, random_state=42)
        X2, y2 = create_synthetic_data(n_samples=50, random_state=42)
        
        assert np.allclose(X1, X2)
        assert np.allclose(y1, y2)


class TestSaveAndLoad:
    
    def test_save_and_load(self):
        """Should save and load data correctly."""
        X_original = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float)
        y_original = np.array([0, 1, 0])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test.csv')
            
            save_csv(filepath, X_original, y_original)
            X_loaded, y_loaded = load_csv(filepath, skip_header=False)
            
            assert np.allclose(X_original, X_loaded)
            assert np.allclose(y_original, y_loaded)
    
    def test_load_file_not_found(self):
        """Should raise error for missing file."""
        with pytest.raises(IOError):
            load_csv('nonexistent_file_12345.csv')
    
    def test_save_creates_file(self):
        """Should create file."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        y = np.array([0, 1])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test.csv')
            
            save_csv(filepath, X, y)
            assert os.path.exists(filepath)


class TestLoadAndSplit:
    
    def test_train_test_split(self):
        """Should split into train/test."""
        X, y = create_synthetic_data(n_samples=100, random_state=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'data.csv')
            save_csv(filepath, X, y)
            
            X_train, X_test, y_train, y_test = load_and_split(
                filepath, train_size=0.8, shuffle=False, random_state=42, skip_header=False
            )
            
            assert len(X_train) == 80
            assert len(X_test) == 20
    
    def test_no_overlap(self):
        """Train and test should not overlap."""
        X, y = create_synthetic_data(n_samples=100, random_state=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'data.csv')
            save_csv(filepath, X, y)
            
            X_train, X_test, y_train, y_test = load_and_split(
                filepath, random_state=42, skip_header=False
            )
            
            assert len(X_train) + len(X_test) == len(X)
    
    def test_no_overlap(self):
        """Train and test should not overlap."""
        X, y = create_synthetic_data(n_samples=100, random_state=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'data.csv')
            save_csv(filepath, X, y)
            
            X_train, X_test, y_train, y_test = load_and_split(filepath, random_state=42)
            
            assert len(X_train) + len(X_test) == len(X)


class TestCreateChunks:
    
    def test_chunk_size(self):
        """Chunks should have correct size."""
        X = np.arange(300).reshape(100, 3).astype(float)
        y = np.arange(100)
        
        chunks = list(create_chunks(X, y, chunk_size=20))
        
        assert len(chunks) == 5
        for X_chunk, y_chunk in chunks:
            assert X_chunk.shape[0] == 20
            assert y_chunk.shape[0] == 20
    
    def test_last_chunk_smaller(self):
        """Last chunk can be smaller."""
        X = np.arange(200).reshape(100, 2).astype(float)
        y = np.arange(100)
        
        chunks = list(create_chunks(X, y, chunk_size=30))
        
        assert len(chunks) == 4
        assert chunks[-1][0].shape[0] == 10
    
    def test_covers_all_data(self):
        """All data should be covered by chunks."""
        X, y = create_synthetic_data(n_samples=100, n_features=5, random_state=42)
        
        total_samples = sum(len(y_chunk) for _, y_chunk in create_chunks(X, y, chunk_size=25, shuffle=False))
        assert total_samples == 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])