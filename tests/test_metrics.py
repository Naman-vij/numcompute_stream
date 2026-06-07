"""Unit tests for metrics.py"""

import numpy as np
import pytest
from numcompute_stream.metrics import (
    StreamingAccuracy, StreamingConfusionMatrix, StreamingPrecision,
    StreamingRecall, StreamingF1, accuracy, confusion_matrix
)


class TestStreamingAccuracy:
    
    def test_perfect_predictions(self):
        metric = StreamingAccuracy()
        y_true = np.array([0, 1, 2, 1])
        y_pred = np.array([0, 1, 2, 1])
        
        metric.update(y_true, y_pred)
        assert metric.result() == 1.0
    
    def test_zero_predictions(self):
        metric = StreamingAccuracy()
        y_true = np.array([0, 1, 2])
        y_pred = np.array([1, 2, 0])
        
        metric.update(y_true, y_pred)
        assert metric.result() == 0.0
    
    def test_partial_accuracy(self):
        metric = StreamingAccuracy()
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        
        metric.update(y_true, y_pred)
        assert metric.result() == 0.75
    
    def test_streaming_accumulation(self):
        metric = StreamingAccuracy()
        
        metric.update(np.array([0, 1]), np.array([0, 1]))
        metric.update(np.array([0, 1]), np.array([0, 1]))
        
        assert metric.result() == 1.0
    
    def test_reset(self):
        metric = StreamingAccuracy()
        metric.update(np.array([0, 1]), np.array([0, 1]))
        metric.reset()
        assert metric.result() == 0.0


class TestStreamingConfusionMatrix:
    
    def test_binary_classification(self):
        metric = StreamingConfusionMatrix(n_classes=2)
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        
        metric.update(y_true, y_pred)
        cm = metric.result()
        
        assert cm[0, 0] == 1
        assert cm[0, 1] == 1
        assert cm[1, 0] == 0
        assert cm[1, 1] == 2
    
    def test_multiclass(self):
        metric = StreamingConfusionMatrix(n_classes=3)
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 1, 1])
        
        metric.update(y_true, y_pred)
        cm = metric.result()
        
        assert cm.shape == (3, 3)
        assert cm.sum() == 5


class TestStreamingPrecision:
    
    def test_perfect_precision(self):
        metric = StreamingPrecision(n_classes=2)
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0, 1, 1])
        
        metric.update(y_true, y_pred)
        assert metric.result() == 1.0
    
    def test_streaming_precision(self):
        metric = StreamingPrecision(n_classes=2)
        
        metric.update(np.array([0, 1]), np.array([0, 1]))
        metric.update(np.array([0, 1]), np.array([0, 1]))
        
        assert metric.result() == 1.0


class TestStreamingRecall:
    
    def test_perfect_recall(self):
        metric = StreamingRecall(n_classes=2)
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0, 1, 1])
        
        metric.update(y_true, y_pred)
        assert metric.result() == 1.0


class TestStreamingF1:
    
    def test_perfect_f1(self):
        metric = StreamingF1(n_classes=2)
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0, 1, 1])
        
        metric.update(y_true, y_pred)
        assert metric.result() == 1.0


class TestBatchMetrics:
    
    def test_accuracy_function(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        
        acc = accuracy(y_true, y_pred)
        assert acc == 0.75
    
    def test_confusion_matrix_function(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2)
        assert cm.sum() == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])