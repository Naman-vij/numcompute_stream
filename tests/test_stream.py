"""Unit tests for stream.py"""

import numpy as np
import pytest
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.metrics import StreamingAccuracy, StreamingPrecision


class TestStreamTrainer:
    
    def setup_method(self):
        """Create test data and trainer."""
        self.X1 = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
        self.y1 = np.array([0, 1, 0])
        
        self.X2 = np.array([[3, 3], [4, 4], [5, 5]], dtype=float)
        self.y2 = np.array([1, 0, 1])
        
        self.X_test = np.array([[1.5, 1.5], [3.5, 3.5]], dtype=float)
        self.y_test = np.array([1, 0])
        
        # Create pipeline
        self.pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', DecisionTreeClassifier(max_depth=3))
        ])
        
        # Create metrics
        self.metrics = {
            'accuracy': StreamingAccuracy(),
        }
    
    def test_trainer_creation(self):
        """Should create trainer."""
        trainer = StreamTrainer(self.pipe, self.metrics)
        assert trainer is not None
    
    def test_fit_chunk(self):
        """Should train on single chunk."""
        trainer = StreamTrainer(self.pipe, self.metrics)
        
        results = trainer.fit_chunk(self.X1, self.y1)
        
        assert 'accuracy' in results
        assert 'time' in results
        assert 0 <= results['accuracy'] <= 1
    
    def test_fit_multiple_chunks(self):
        """Should train on multiple chunks."""
        trainer = StreamTrainer(self.pipe, self.metrics)
        
        results1 = trainer.fit_chunk(self.X1, self.y1)
        results2 = trainer.fit_chunk(self.X2, self.y2)
        
        assert 'accuracy' in results1
        assert 'accuracy' in results2
    
    def test_score_chunk(self):
        """Should score without training."""
        trainer = StreamTrainer(self.pipe, self.metrics)
        
        trainer.fit_chunk(self.X1, self.y1)
        trainer.fit_chunk(self.X2, self.y2)
        
        results = trainer.score_chunk(self.X_test, self.y_test)
        
        assert 'accuracy' in results
        assert 0 <= results['accuracy'] <= 1
    
    def test_metric_history(self):
        """Should track metric history."""
        trainer = StreamTrainer(self.pipe, self.metrics)
        
        trainer.fit_chunk(self.X1, self.y1)
        trainer.fit_chunk(self.X2, self.y2)
        
        history = trainer.get_metric_history()
        
        assert 'accuracy' in history
        assert len(history['accuracy']) == 2
    
    def test_reset_metrics(self):
        """Should reset metrics."""
        trainer = StreamTrainer(self.pipe, self.metrics)
        
        trainer.fit_chunk(self.X1, self.y1)
        trainer.reset_metrics()
        
        history = trainer.get_metric_history()
        assert len(history['accuracy']) == 0
    
    def test_get_stats(self):
        """Should return training statistics."""
        trainer = StreamTrainer(self.pipe, self.metrics)
        
        trainer.fit_chunk(self.X1, self.y1)
        trainer.fit_chunk(self.X2, self.y2)
        
        stats = trainer.get_stats()
        
        assert stats['total_samples'] == 6
        assert stats['chunk_count'] == 2
        assert stats['avg_time_per_chunk'] >= 0
        assert 'accuracy' in stats['current_metrics']
    
    def test_multiple_metrics(self):
        """Should track multiple metrics."""
        metrics = {
            'accuracy': StreamingAccuracy(),
            'precision': StreamingPrecision(n_classes=2)
        }
        
        trainer = StreamTrainer(self.pipe, metrics)
        
        results = trainer.fit_chunk(self.X1, self.y1)
        
        assert 'accuracy' in results
        assert 'precision' in results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])