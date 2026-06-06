"""Unit tests for tree.py"""

import numpy as np
import pytest
from numcompute_stream.tree import DecisionTreeClassifier


class TestDecisionTree:
    
    def setup_method(self):
        """Create test dataset."""
        self.X_train = np.array([
            [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]
        ], dtype=float)
        self.y_train = np.array([0, 1, 1, 0, 1, 0])
    
    def test_fit_and_predict(self):
        tree = DecisionTreeClassifier(max_depth=5)
        tree.fit(self.X_train, self.y_train)
        
        y_pred = tree.predict(self.X_train)
        assert y_pred.shape == self.y_train.shape
    
    def test_predict_shape(self):
        tree = DecisionTreeClassifier(max_depth=5)
        tree.fit(self.X_train, self.y_train)
        
        y_pred = tree.predict(self.X_train)
        assert y_pred.shape == (6,)
    
    def test_score(self):
        tree = DecisionTreeClassifier(max_depth=5)
        tree.fit(self.X_train, self.y_train)
        
        acc = tree.score(self.X_train, self.y_train)
        assert 0 <= acc <= 1
    
    def test_partial_fit_streaming(self):
        tree = DecisionTreeClassifier(max_depth=5)
        
        tree.partial_fit(self.X_train[:3], self.y_train[:3])
        acc1 = tree.score(self.X_train[:3], self.y_train[:3])
        
        tree.partial_fit(self.X_train[3:], self.y_train[3:])
        acc2 = tree.score(self.X_train, self.y_train)
        
        assert 0 <= acc1 <= 1
        assert 0 <= acc2 <= 1
    
    def test_max_depth_limit(self):
        tree1 = DecisionTreeClassifier(max_depth=1)
        tree2 = DecisionTreeClassifier(max_depth=10)
        
        tree1.fit(self.X_train, self.y_train)
        tree2.fit(self.X_train, self.y_train)
        
        assert tree1.tree is not None
        assert tree2.tree is not None
    
    def test_gini_criterion(self):
        tree = DecisionTreeClassifier(criterion='gini', max_depth=5)
        tree.fit(self.X_train, self.y_train)
        
        y_pred = tree.predict(self.X_train)
        assert y_pred.shape == (6,)
    
    def test_entropy_criterion(self):
        tree = DecisionTreeClassifier(criterion='entropy', max_depth=5)
        tree.fit(self.X_train, self.y_train)
        
        y_pred = tree.predict(self.X_train)
        assert y_pred.shape == (6,)
    
    def test_single_sample(self):
        tree = DecisionTreeClassifier()
        tree.fit(self.X_train, self.y_train)
        
        y_pred = tree.predict(self.X_train[[0]])
        assert y_pred.shape == (1,)
    
    def test_with_nan_values(self):
        X_nan = self.X_train.copy()
        X_nan[0, 0] = np.nan
        
        tree = DecisionTreeClassifier()
        tree.fit(X_nan, self.y_train)
        
        y_pred = tree.predict(X_nan)
        assert y_pred.shape == (6,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])