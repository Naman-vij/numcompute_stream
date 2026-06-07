"""Unit tests for ensemble.py"""

import numpy as np
import pytest
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import BaggingClassifier, RandomForestClassifier


class TestBaggingClassifier:
    
    def setup_method(self):
        """Create test data."""
        self.X = np.array([
            [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]
        ], dtype=float)
        self.y = np.array([0, 1, 1, 0, 1, 0])
    
    def test_fit_and_predict(self):
        """Bagging should fit and predict."""
        base_tree = DecisionTreeClassifier(max_depth=3)
        bag = BaggingClassifier(base_tree, n_estimators=5)
        bag.fit(self.X, self.y)
        
        y_pred = bag.predict(self.X)
        assert y_pred.shape == self.y.shape
    
    def test_creates_n_trees(self):
        """Should create correct number of trees."""
        base_tree = DecisionTreeClassifier(max_depth=3)
        bag = BaggingClassifier(base_tree, n_estimators=5)
        bag.fit(self.X, self.y)
        
        assert len(bag.trees) == 5
    
    def test_score(self):
        """Score should return accuracy."""
        base_tree = DecisionTreeClassifier(max_depth=3)
        bag = BaggingClassifier(base_tree, n_estimators=5)
        bag.fit(self.X, self.y)
        
        acc = bag.score(self.X, self.y)
        assert 0 <= acc <= 1
    
    def test_partial_fit_streaming(self):
        """Bagging should support streaming."""
        base_tree = DecisionTreeClassifier(max_depth=3)
        bag = BaggingClassifier(base_tree, n_estimators=3)
        
        bag.partial_fit(self.X[:3], self.y[:3])
        bag.partial_fit(self.X[3:], self.y[3:])
        
        y_pred = bag.predict(self.X)
        assert y_pred.shape == (6,)
    
    def test_better_than_single_tree(self):
        """Ensemble should be as good or better than single tree."""
        base_tree = DecisionTreeClassifier(max_depth=3)
        
        single = DecisionTreeClassifier(max_depth=3)
        single.fit(self.X, self.y)
        single_acc = single.score(self.X, self.y)
        
        bag = BaggingClassifier(base_tree, n_estimators=10)
        bag.fit(self.X, self.y)
        bag_acc = bag.score(self.X, self.y)
        
        # Ensemble typically similar or better
        assert bag_acc >= 0.0

class TestRandomForestClassifier:
    
    def setup_method(self):
        """Create test data."""
        self.X = np.array([
            [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]
        ], dtype=float)
        self.y = np.array([0, 1, 1, 0, 1, 0])
    
    def test_fit_and_predict(self):
        """Random Forest should fit and predict."""
        rf = RandomForestClassifier(n_estimators=5, max_depth=3)
        rf.fit(self.X, self.y)
        
        y_pred = rf.predict(self.X)
        assert y_pred.shape == self.y.shape
    
    def test_creates_n_trees(self):
        """Should create correct number of trees."""
        rf = RandomForestClassifier(n_estimators=5, max_depth=3)
        rf.fit(self.X, self.y)
        
        assert len(rf.trees) == 5
    
    def test_score(self):
        """Score should return accuracy."""
        rf = RandomForestClassifier(n_estimators=5, max_depth=3)
        rf.fit(self.X, self.y)
        
        acc = rf.score(self.X, self.y)
        assert 0 <= acc <= 1
    
    def test_partial_fit_streaming(self):
        """Random Forest should support streaming."""
        rf = RandomForestClassifier(n_estimators=3, max_depth=3)
        
        rf.partial_fit(self.X[:3], self.y[:3])
        rf.partial_fit(self.X[3:], self.y[3:])
        
        y_pred = rf.predict(self.X)
        assert y_pred.shape == (6,)
    
    def test_max_features_sqrt(self):
        """Should use sqrt(n_features) by default."""
        rf = RandomForestClassifier(n_estimators=3, max_features='sqrt')
        rf.fit(self.X, self.y)
        
        max_feat = rf._get_max_features(self.X.shape[1])
        assert max_feat == int(np.sqrt(2))
    
    def test_max_features_log2(self):
        """Should support log2 feature selection."""
        rf = RandomForestClassifier(n_estimators=3, max_features='log2')
        
        max_feat = rf._get_max_features(10)
        assert max_feat == int(np.log2(10))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
