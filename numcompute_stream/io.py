"""
io.py - Input/Output Utilities

Handle loading and saving data from/to files.
Create synthetic datasets for testing.
"""

import numpy as np
from typing import Tuple, Optional


def load_csv(filepath: str, sep: str = ',', skip_header: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from CSV file.
    
    Assumes:
    - Last column is the label (y)
    - All other columns are features (X)
    - Numeric data only
    
    Args:
        filepath: Path to CSV file
        sep: Delimiter character
        skip_header: Whether to skip first row (header)
    
    Returns:
        (X, y) where X is features and y is labels
    
    Example:
        X, y = load_csv('data/iris.csv')
        print(X.shape)  # (150, 4)
        print(y.shape)  # (150,)
    
    Raises:
        IOError: If file cannot be read
    """
    try:
        skip_rows = 1 if skip_header else 0
        data = np.genfromtxt(filepath, delimiter=sep, skip_header=skip_rows)
        
        if data.ndim == 1:
            raise ValueError("Data must have at least 2 columns (features + labels)")
        
        X = data[:, :-1]
        y = data[:, -1].astype(int)
        
        return X, y
    
    except FileNotFoundError:
        raise IOError(f"File not found: {filepath}")
    except Exception as e:
        raise IOError(f"Could not load CSV: {e}")


def save_csv(filepath: str, X: np.ndarray, y: np.ndarray, sep: str = ',',
             header: Optional[str] = None) -> None:
    """
    Save features and labels to CSV file.
    
    Args:
        filepath: Path to save file
        X: Features, shape (n_samples, n_features)
        y: Labels, shape (n_samples,)
        sep: Delimiter character
        header: Optional header row (comma-separated)
    
    Example:
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        save_csv('output.csv', X, y)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    if len(X) != len(y):
        raise ValueError(f"X and y must have same number of samples: {len(X)} vs {len(y)}")
    
    data = np.column_stack([X, y])
    
    try:
        if header is not None:
            np.savetxt(filepath, data, delimiter=sep, header=header, comments='')
        else:
            np.savetxt(filepath, data, delimiter=sep)
    except Exception as e:
        raise IOError(f"Could not save CSV: {e}")


def create_synthetic_data(n_samples: int = 1000, n_features: int = 10,
                         n_classes: int = 2, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic classification dataset for testing.
    
    Generates:
    - Random normal features
    - Labels based on linear combinations of features
    - Reproducible with random_state
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features per sample
        n_classes: Number of classes
        random_state: Random seed for reproducibility
    
    Returns:
        (X, y) synthetic features and labels
    
    Example:
        # Generate binary classification dataset
        X, y = create_synthetic_data(n_samples=500, n_features=5, n_classes=2)
        print(X.shape)  # (500, 5)
        print(np.unique(y))  # [0, 1]
        
        # Generate multiclass dataset
        X, y = create_synthetic_data(n_samples=1000, n_features=10, n_classes=3)
        print(np.unique(y))  # [0, 1, 2]
    """
    rng = np.random.RandomState(random_state)
    
    # Generate random features (mean=0, std=1)
    X = rng.randn(n_samples, n_features)
    
    # Create labels based on linear combination of features
    # Different thresholds for different classes
    y = np.zeros(n_samples, dtype=int)
    
    for k in range(n_classes):
        # Create boundary: sum of first 2 features
        threshold = 2 * k - n_classes
        mask = (X[:, 0] + X[:, 1]) > threshold
        y[mask] = k
    
    # Ensure all classes are represented
    y = np.clip(y, 0, n_classes - 1)
    
    return X, y


def load_and_split(filepath: str, train_size: float = 0.8,
                   shuffle: bool = True, random_state: Optional[int] = None,
                   skip_header: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CSV and split into train/test sets.
    
    Args:
        filepath: Path to CSV file
        train_size: Fraction for training (0.0 to 1.0)
        shuffle: Whether to shuffle before splitting
        random_state: Random seed
        skip_header: Whether to skip first row
    
    Returns:
        (X_train, X_test, y_train, y_test)
    
    Example:
        X_train, X_test, y_train, y_test = load_and_split('data.csv', train_size=0.8)
        print(X_train.shape)  # 80% of data
        print(X_test.shape)   # 20% of data
    """
    X, y = load_csv(filepath, skip_header=skip_header)
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    rng = np.random.RandomState(random_state)
    if shuffle:
        rng.shuffle(indices)
    
    split_idx = int(n_samples * train_size)
    
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    return X_train, X_test, y_train, y_test


def create_chunks(X: np.ndarray, y: np.ndarray, chunk_size: int,
                  shuffle: bool = False, random_state: Optional[int] = None):
    """
    Split data into chunks for streaming.
    
    Yields chunks of specified size for iterative training.
    
    Args:
        X: Features, shape (n_samples, n_features)
        y: Labels, shape (n_samples,)
        chunk_size: Size of each chunk
        shuffle: Whether to shuffle before chunking
        random_state: Random seed
    
    Yields:
        (X_chunk, y_chunk) tuples
    
    Example:
        for X_chunk, y_chunk in create_chunks(X, y, chunk_size=100):
            model.partial_fit(X_chunk, y_chunk)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)
    
    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        chunk_indices = indices[start_idx:end_idx]
        
        X_chunk = X[chunk_indices]
        y_chunk = y[chunk_indices]
        
        yield X_chunk, y_chunk