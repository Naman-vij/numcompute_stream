"""
metrics.py - Classification Metrics with Streaming Support

Implements metrics that accumulate over chunks for streaming evaluation.
"""

import numpy as np
from typing import Optional


class StreamingMetric:
    """Base class for streaming metrics."""
    
    def update(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Update metric with new predictions."""
        raise NotImplementedError
    
    def result(self) -> float:
        """Return current metric value."""
        raise NotImplementedError
    
    def reset(self):
        """Reset metric to initial state."""
        raise NotImplementedError


class StreamingAccuracy(StreamingMetric):
    """
    Compute accuracy incrementally.
    
    Tracks: correct predictions / total predictions
    """
    
    def __init__(self):
        self.correct = 0
        self.total = 0
    
    def update(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Update with new chunk."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        self.correct += np.sum(y_true == y_pred)
        self.total += len(y_true)
    
    def result(self) -> float:
        """Return current accuracy."""
        return self.correct / self.total if self.total > 0 else 0.0
    
    def reset(self):
        """Reset counters."""
        self.correct = 0
        self.total = 0


class StreamingConfusionMatrix(StreamingMetric):
    """
    Compute confusion matrix incrementally.
    
    Shows: True Positives, False Positives, True Negatives, False Negatives
    """
    
    def __init__(self, n_classes: Optional[int] = None):
        self.n_classes = n_classes
        self.matrix = None
    
    def update(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Update confusion matrix."""
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        
        if self.n_classes is None:
            self.n_classes = max(y_true.max(), y_pred.max()) + 1
            self.matrix = np.zeros((self.n_classes, self.n_classes), dtype=int)
        
        for i in range(len(y_true)):
            if 0 <= y_true[i] < self.n_classes and 0 <= y_pred[i] < self.n_classes:
                self.matrix[y_true[i], y_pred[i]] += 1
    
    def result(self) -> np.ndarray:
        """Return current confusion matrix."""
        return self.matrix if self.matrix is not None else np.array([])
    
    def reset(self):
        """Reset matrix."""
        self.matrix = None


class StreamingPrecision(StreamingMetric):
    """
    Compute precision incrementally (macro-average across classes).
    
    Precision = TP / (TP + FP)
    How many predicted positives are actually correct?
    """
    
    def __init__(self, n_classes: Optional[int] = None):
        self.n_classes = n_classes
        self.tp = {}
        self.fp = {}
    
    def update(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Update precision."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if self.n_classes is None:
            self.n_classes = max(y_true.max(), y_pred.max()) + 1
            for c in range(self.n_classes):
                self.tp[c] = 0
                self.fp[c] = 0
        
        for i in range(len(y_true)):
            pred_class = int(y_pred[i])
            true_class = int(y_true[i])
            
            if pred_class == true_class:
                self.tp[pred_class] = self.tp.get(pred_class, 0) + 1
            else:
                self.fp[pred_class] = self.fp.get(pred_class, 0) + 1
    
    def result(self) -> float:
        """Return macro-averaged precision."""
        precisions = []
        for c in range(self.n_classes):
            tp = self.tp.get(c, 0)
            fp = self.fp.get(c, 0)
            
            if tp + fp > 0:
                precisions.append(tp / (tp + fp))
            else:
                precisions.append(0.0)
        
        return np.mean(precisions) if precisions else 0.0
    
    def reset(self):
        """Reset counters."""
        self.tp = {}
        self.fp = {}


class StreamingRecall(StreamingMetric):
    """
    Compute recall incrementally (macro-average across classes).
    
    Recall = TP / (TP + FN)
    How many actual positives did we catch?
    """
    
    def __init__(self, n_classes: Optional[int] = None):
        self.n_classes = n_classes
        self.tp = {}
        self.fn = {}
    
    def update(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Update recall."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if self.n_classes is None:
            self.n_classes = max(y_true.max(), y_pred.max()) + 1
            for c in range(self.n_classes):
                self.tp[c] = 0
                self.fn[c] = 0
        
        for i in range(len(y_true)):
            pred_class = int(y_pred[i])
            true_class = int(y_true[i])
            
            if pred_class == true_class:
                self.tp[true_class] = self.tp.get(true_class, 0) + 1
            else:
                self.fn[true_class] = self.fn.get(true_class, 0) + 1
    
    def result(self) -> float:
        """Return macro-averaged recall."""
        recalls = []
        for c in range(self.n_classes):
            tp = self.tp.get(c, 0)
            fn = self.fn.get(c, 0)
            
            if tp + fn > 0:
                recalls.append(tp / (tp + fn))
            else:
                recalls.append(0.0)
        
        return np.mean(recalls) if recalls else 0.0
    
    def reset(self):
        """Reset counters."""
        self.tp = {}
        self.fn = {}


class StreamingF1(StreamingMetric):
    """
    Compute F1 score incrementally.
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    Balance between precision and recall
    """
    
    def __init__(self, n_classes: Optional[int] = None):
        self.precision = StreamingPrecision(n_classes)
        self.recall = StreamingRecall(n_classes)
    
    def update(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Update F1 score."""
        self.precision.update(y_true, y_pred)
        self.recall.update(y_true, y_pred)
    
    def result(self) -> float:
        """Return F1 score."""
        p = self.precision.result()
        r = self.recall.result()
        
        if p + r == 0:
            return 0.0
        
        return 2 * (p * r) / (p + r)
    
    def reset(self):
        """Reset both precision and recall."""
        self.precision.reset()
        self.recall.reset()


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy for batch data."""
    return np.mean(np.asarray(y_true) == np.asarray(y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: Optional[int] = None) -> np.ndarray:
    """Compute confusion matrix for batch data."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    
    if n_classes is None:
        n_classes = max(y_true.max(), y_pred.max()) + 1
    
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y_true)):
        if 0 <= y_true[i] < n_classes and 0 <= y_pred[i] < n_classes:
            matrix[y_true[i], y_pred[i]] += 1
    
    return matrix


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro-averaged precision for batch data."""
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    
    precisions = np.divide(tp, tp + fp, where=(tp + fp) != 0, 
                          out=np.zeros_like(tp, dtype=float))
    return np.mean(precisions)


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro-averaged recall for batch data."""
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    
    recalls = np.divide(tp, tp + fn, where=(tp + fn) != 0,
                       out=np.zeros_like(tp, dtype=float))
    return np.mean(recalls)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro-averaged F1 score for batch data."""
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    
    if p + r == 0:
        return 0.0
    
    return 2 * (p * r) / (p + r)