"""Unit tests for pipeline.py"""

import numpy as np
import pytest
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler, SimpleImputer
from numcompute_stream.tree import DecisionTreeClassifier


class TestPipeline:
    
    def setup_method(self):
        """Create test data."""
        self.X = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0]
        ], dtype=float)
        self.y = np.array([0, 1, 1, 0])
    
    def test_pipeline_creation(self):
        """Should create pipeline."""
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', DecisionTreeClassifier())
        ])
        assert pipe is not None
    
    def test_pipeline_fit(self):
        """Should fit pipeline."""
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', DecisionTreeClassifier())
        ])
        pipe.fit(self.X, self.y)
        assert pipe.model.tree is not None
    
    def test_pipeline_predict(self):
        """Should predict with pipeline."""
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', DecisionTreeClassifier())
        ])
        pipe.fit(self.X, self.y)
        
        y_pred = pipe.predict(self.X)
        assert y_pred.shape == self.y.shape
    
    def test_pipeline_score(self):
        """Should compute score."""
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', DecisionTreeClassifier())
        ])
        pipe.fit(self.X, self.y)
        
        score = pipe.score(self.X, self.y)
        assert 0 <= score <= 1
    
    def test_pipeline_partial_fit(self):
        """Should support streaming."""
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', DecisionTreeClassifier())
        ])
        
        # First partial_fit initializes scaler
        pipe.partial_fit(self.X[:2], self.y[:2])
        
        # Second partial_fit updates scaler and model
        pipe.partial_fit(self.X[2:], self.y[2:])
        
        # Should be able to predict
        y_pred = pipe.predict(self.X)
        assert y_pred.shape == (4,)
        assert len(np.unique(y_pred)) > 0
    
    def test_pipeline_multiple_transformers(self):
        """Should chain multiple transformers."""
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', DecisionTreeClassifier())
        ])
        pipe.fit(self.X, self.y)
        
        y_pred = pipe.predict(self.X)
        assert y_pred.shape == self.y.shape
    
    def test_pipeline_with_nan(self):
        """Should handle NaN in pipeline."""
        X_nan = self.X.copy()
        X_nan[0, 0] = np.nan
        
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', DecisionTreeClassifier())
        ])
        pipe.fit(X_nan, self.y)
        
        y_pred = pipe.predict(X_nan)
        assert y_pred.shape == self.y.shape
    
    def test_pipeline_error_empty(self):
        """Should raise error for empty pipeline."""
        with pytest.raises(ValueError):
            Pipeline([])
    
    def test_pipeline_error_no_predict(self):
        """Should raise error if last step has no predict."""
        with pytest.raises(ValueError):
            Pipeline([
                ('scaler', StandardScaler())
            ])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])