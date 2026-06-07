"""
ensemble.py - Ensemble Methods with Streaming Support

Implements Bagging and Random Forest using multiple decision trees.
"""

import numpy as np
from typing import Optional, List
from collections import Counter


class BaggingClassifier:
    """
    Bagging (Bootstrap Aggregating) Ensemble.
    
    How it works:
    1. Create N trees
    2. Each tree trained on random sample (WITH replacement)
    3. Predict: Majority voting across all trees
    
    Why it works:
    - Different samples = different trees
    - Diversity reduces overfitting
    - Majority vote is more robust
    
    Parameters:
        base_estimator: Tree class to use
        n_estimators: Number of trees
        max_samples: Size of bootstrap sample
        random_state: Random seed
    """
    
    def __init__(self, base_estimator, n_estimators: int = 10,
                 max_samples: Optional[int] = None, random_state: Optional[int] = None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        
        self.trees = []
        self.n_samples_seen = 0
        self.rng = np.random.RandomState(random_state)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaggingClassifier':
        """Train ensemble on full dataset."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = X.shape[0]
        max_samples = self.max_samples or n_samples
        
        self.trees = []
        
        # Create N trees
        for _ in range(self.n_estimators):
            # Bootstrap sample: random sample WITH replacement
            indices = self.rng.choice(n_samples, size=max_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            
            # Train tree on this bootstrap sample
            tree = self.base_estimator.__class__(
    max_depth=self.base_estimator.max_depth,
    min_samples_split=self.base_estimator.min_samples_split,
    criterion=self.base_estimator.criterion,
    random_state=self.rng.randint(0, 2**31)
)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        self.n_samples_seen = n_samples
        return self
    
    def partial_fit(self, X_chunk: np.ndarray, y_chunk: np.ndarray) -> 'BaggingClassifier':
        """Update ensemble with new chunk (streaming)."""
        X_chunk = np.asarray(X_chunk)
        y_chunk = np.asarray(y_chunk)
        
        if X_chunk.ndim == 1:
            X_chunk = X_chunk.reshape(-1, 1)
        
        # Initialize trees if needed
        if len(self.trees) == 0:
            for _ in range(self.n_estimators):
                tree = self.base_estimator.__class__(
    max_depth=self.base_estimator.max_depth,
    min_samples_split=self.base_estimator.min_samples_split,
    criterion=self.base_estimator.criterion,
    random_state=self.rng.randint(0, 2**31)
)
                self.trees.append(tree)
        
        # Update each tree with new chunk
        for tree in self.trees:
            tree.partial_fit(X_chunk, y_chunk)
        
        self.n_samples_seen += X_chunk.shape[0]
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict by majority voting.
        
        How it works:
        1. Get prediction from each tree
        2. Count votes
        3. Pick class with most votes
        
        Example:
        Tree 1 predicts: 0
        Tree 2 predicts: 1
        Tree 3 predicts: 0
        Result: 0 (2 votes)
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Majority voting
        y_pred = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            y_pred[i] = Counter(votes).most_common(1)[0][0]
        
        return y_pred
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class RandomForestClassifier:
    """
    Random Forest Ensemble.
    
    Bagging + Random Feature Selection at Each Split
    
    How it works:
    1. Create N trees (same as Bagging)
    2. Each tree trained on bootstrap sample
    3. At each split: Only consider max_features random features
    4. Predict: Majority voting
    
    Why better than Bagging:
    - Feature randomization adds diversity
    - Trees are more independent
    - Better generalization
    
    Parameters:
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        max_features: Features to consider per split ('sqrt', 'log2', or int)
        random_state: Random seed
    """
    
    def __init__(self, n_estimators: int = 10, max_depth: int = 10,
                 max_features: str = 'sqrt', random_state: Optional[int] = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        
        self.trees = []
        self.n_samples_seen = 0
        self.n_features_in = None
        self.rng = np.random.RandomState(random_state)
    
    def _get_max_features(self, n_features: int) -> int:
        """Calculate number of features to use per split."""
        if isinstance(self.max_features, int):
            return self.max_features
        elif self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        else:
            return n_features
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClassifier':
        """Train random forest on full dataset."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_features_in = X.shape[1]
        n_samples = X.shape[0]
        
        from .tree import DecisionTreeClassifier
        
        self.trees = []
        
        # Create N trees
        for _ in range(self.n_estimators):
            # Bootstrap sample
            indices = self.rng.choice(n_samples, size=n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            
            # Create tree with feature randomization
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=self.rng.randint(0, 2**31)
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        self.n_samples_seen = n_samples
        return self
    
    def partial_fit(self, X_chunk: np.ndarray, y_chunk: np.ndarray) -> 'RandomForestClassifier':
        """Update forest with new chunk (streaming)."""
        X_chunk = np.asarray(X_chunk)
        y_chunk = np.asarray(y_chunk)
        
        if X_chunk.ndim == 1:
            X_chunk = X_chunk.reshape(-1, 1)
        
        if self.n_features_in is None:
            self.n_features_in = X_chunk.shape[1]
        
        # Initialize trees if needed
        if len(self.trees) == 0:
            from .tree import DecisionTreeClassifier
            
            for _ in range(self.n_estimators):
                tree = DecisionTreeClassifier(
                    max_depth=self.max_depth,
                    random_state=self.rng.randint(0, 2**31)
                )
                self.trees.append(tree)
        
        # Update each tree with new chunk
        for tree in self.trees:
            tree.partial_fit(X_chunk, y_chunk)
        
        self.n_samples_seen += X_chunk.shape[0]
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict by majority voting."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Majority voting
        y_pred = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            y_pred[i] = Counter(votes).most_common(1)[0][0]
        
        return y_pred
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)