"""
tree.py - Decision Tree Classifier with Streaming Support

Implements a depth-limited decision tree with Gini/Entropy-based splitting.
Supports incremental training via partial_fit().
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from collections import Counter


class Node:
    """Represents a node in the decision tree."""
    
    def __init__(self, feature: Optional[int] = None, threshold: Optional[float] = None,
                 left=None, right=None, value: Optional[int] = None):
        """
        Args:
            feature: Feature index for splitting (None if leaf)
            threshold: Split threshold (None if leaf)
            left: Left child node
            right: Right child node
            value: Class label (if leaf node)
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.samples = 0


class DecisionTreeClassifier:
    """
    Decision Tree Classifier with streaming support.
    
    Parameters:
        max_depth: Maximum tree depth (default: 10)
        min_samples_split: Minimum samples to split a node (default: 2)
        criterion: 'gini' or 'entropy' for split criterion
        random_state: Random seed for reproducibility
    """
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2,
                 criterion: str = 'gini', random_state: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.random_state = random_state
        self.tree = None
        self.n_classes = None
        self.n_features = None
        self.classes = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeClassifier':
        """
        Build tree from data (batch learning).
        
        Args:
            X: Features, shape (n_samples, n_features)
            y: Labels, shape (n_samples,)
        
        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_features = X.shape[1]
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        
        self.tree = self._grow_tree(X, y, depth=0)
        return self
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes: Optional[np.ndarray] = None) -> 'DecisionTreeClassifier':
        """
        Incrementally update tree with new chunk of data.
        
        Args:
            X: New feature chunk, shape (n_samples, n_features)
            y: New labels, shape (n_samples,)
            classes: All possible classes (if first time)
        
        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if self.tree is None:
            if classes is not None:
                self.classes = np.asarray(classes)
            else:
                self.classes = np.unique(y)
            
            self.n_classes = len(self.classes)
            self.n_features = X.shape[1]
            self.tree = self._grow_tree(X, y, depth=0)
        else:
            self.tree = self._update_tree(self.tree, X, y, depth=0)
        
        return self
    
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """
        Recursively grow tree (batch mode).
        """
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))
        
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            majority_class = Counter(y).most_common(1)[0][0]
            node = Node(value=majority_class)
            node.samples = n_samples
            return node
        
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:
            majority_class = Counter(y).most_common(1)[0][0]
            node = Node(value=majority_class)
            node.samples = n_samples
            return node
        
        mask = X[:, best_feature] <= best_threshold
        X_left, y_left = X[mask], y[mask]
        X_right, y_right = X[~mask], y[~mask]
        
        node = Node(feature=best_feature, threshold=best_threshold)
        node.left = self._grow_tree(X_left, y_left, depth + 1)
        node.right = self._grow_tree(X_right, y_right, depth + 1)
        node.samples = n_samples
        
        return node
    
    def _update_tree(self, node: Node, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """
        Update existing tree with new data (streaming mode).
        """
        if node.value is not None:
            if depth < self.max_depth and len(np.unique(y)) > 1:
                best_feature, best_threshold = self._find_best_split(X, y)
                if best_feature is not None:
                    mask = X[:, best_feature] <= best_threshold
                    if mask.sum() > 0 and (~mask).sum() > 0:
                        node.feature = best_feature
                        node.threshold = best_threshold
                        node.value = None
                        X_left, y_left = X[mask], y[mask]
                        X_right, y_right = X[~mask], y[~mask]
                        node.left = self._grow_tree(X_left, y_left, depth + 1)
                        node.right = self._grow_tree(X_right, y_right, depth + 1)
            node.samples += len(y)
            return node
        
        mask = X[:, node.feature] <= node.threshold
        if mask.sum() > 0:
            node.left = self._update_tree(node.left, X[mask], y[mask], depth + 1)
        if (~mask).sum() > 0:
            node.right = self._update_tree(node.right, X[~mask], y[~mask], depth + 1)
        
        node.samples += len(y)
        return node
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """
        Find best feature and threshold for splitting.
        """
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        parent_impurity = self._impurity(y)
        n_samples = len(y)
        
        for feature_idx in range(X.shape[1]):
            feature_vals = X[:, feature_idx]
            
            valid_vals = feature_vals[~np.isnan(feature_vals)]
            if len(valid_vals) == 0 or len(np.unique(valid_vals)) == 1:
                continue
            
            thresholds = np.unique(valid_vals)
            
            for threshold in thresholds:
                mask = feature_vals <= threshold
                if mask.sum() == 0 or mask.sum() == n_samples:
                    continue
                
                y_left = y[mask]
                y_right = y[~mask]
                
                n_left, n_right = len(y_left), len(y_right)
                impurity_left = self._impurity(y_left)
                impurity_right = self._impurity(y_right)
                
                weighted_impurity = (n_left / n_samples) * impurity_left + \
                                    (n_right / n_samples) * impurity_right
                
                gain = parent_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _impurity(self, y: np.ndarray) -> float:
        """
        Calculate impurity (Gini or Entropy).
        """
        if len(y) == 0:
            return 0.0
        
        counts = np.bincount(y.astype(int), minlength=self.n_classes)
        probs = counts / len(y)
        
        if self.criterion == 'gini':
            return 1.0 - np.sum(probs ** 2)
        elif self.criterion == 'entropy':
            probs = probs[probs > 0]
            return -np.sum(probs * np.log2(probs + 1e-10))
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features, shape (n_samples, n_features)
        
        Returns:
            Predicted labels, shape (n_samples,)
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        return np.array([self._predict_sample(x, self.tree) for x in X])
    
    def _predict_sample(self, x: np.ndarray, node: Node) -> int:
        """Predict single sample by traversing tree."""
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy on test set.
        
        Args:
            X: Features
            y: True labels
        
        Returns:
            Accuracy (0 to 1)
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)