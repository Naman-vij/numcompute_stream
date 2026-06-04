# provides function for online/streaming computation for statistics without storing all data in memory

import numpy as np
from typing import Tuple, Optional

class StreamingMean:
    def __init__(self):
        # compute mean incrementally using welford's algo
        self.mean = 0.0
        self.n = 0

    def update(self, x: np.ndarray):
        # update mean with new values
        if isinstance(x, (int, float)):
            x = np.array([x])

        x = np.asarray(x).flatten()
        for val in x:
            if not np.isnan(val):
                self.n += 1
                delta = val - self.mean
                self.mean += delta / self.n

    def reset(self) -> float:
        # return current mean 
        return self.mean if self.n > 0 else 0.0
    
    def reset(self):
        # reset to initial state
        self.mean = 0.0
        self.n = 0

class StreamingVariance:
    """Compute variance incrementally using Welford's algorithm."""
    
    def __init__(self, ddof: int = 1):
        """
        ddof: Delta Degrees of Freedom (1 for sample variance, 0 for population)
        """
        self.mean = 0.0
        self.M2 = 0.0
        self.n = 0
        self.ddof = ddof
    
    def update(self, x: np.ndarray):
        """Update variance with new value(s)."""
        if isinstance(x, (int, float)):
            x = np.array([x])
        
        x = np.asarray(x).flatten()
        for val in x:
            if not np.isnan(val):
                self.n += 1
                delta = val - self.mean
                self.mean += delta / self.n
                delta2 = val - self.mean
                self.M2 += delta * delta2
    
    def result(self) -> float:
        """Return current variance estimate."""
        if self.n <= self.ddof:
            return 0.0
        return self.M2 / (self.n - self.ddof)
    
    def std(self) -> float:
        """Return current standard deviation estimate."""
        return np.sqrt(max(0.0, self.result()))
    
    def reset(self):
        """Reset to initial state."""
        self.mean = 0.0
        self.M2 = 0.0
        self.n = 0


def streaming_mean_per_feature(X: np.ndarray, means: Optional[np.ndarray] = None, 
                               n: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Compute/update feature-wise means for multiple features.
    
    Args:
        X: Data chunk of shape (n_samples, n_features)
        means: Previous means (if updating), shape (n_features,)
        n: Previous sample count (if updating)
    
    Returns:
        Updated means (n_features,), updated count
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples, n_features = X.shape
    
    if means is None:
        means = np.zeros(n_features)
        n = 0
    else:
        means = np.copy(means)
    
    for i in range(n_samples):
        for j in range(n_features):
            if not np.isnan(X[i, j]):
                n += 1
                delta = X[i, j] - means[j]
                means[j] += delta / n
    
    return means, n


def streaming_var_per_feature(X: np.ndarray, means: np.ndarray, 
                              M2: Optional[np.ndarray] = None,
                              n: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute/update feature-wise variances using Welford's algorithm.
    
    Args:
        X: Data chunk of shape (n_samples, n_features)
        means: Current feature means, shape (n_features,)
        M2: Previous M2 (sum of squared diffs), shape (n_features,)
        n: Previous sample count
    
    Returns:
        Updated M2, updated variances, updated count
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples, n_features = X.shape
    
    if M2 is None:
        M2 = np.zeros(n_features)
        n = 0
    else:
        M2 = np.copy(M2)
    
    means = np.copy(means)
    
    for i in range(n_samples):
        for j in range(n_features):
            if not np.isnan(X[i, j]):
                n += 1
                delta = X[i, j] - means[j]
                means[j] += delta / n
                delta2 = X[i, j] - means[j]
                M2[j] += delta * delta2
    
    variances = np.divide(M2, np.maximum(n - 1, 1))
    return M2, variances, n

