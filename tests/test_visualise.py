"""Unit tests for visualise.py"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from numcompute_stream import visualise
import matplotlib
matplotlib.use('Agg')

class TestPlotMetricOverTime:
    
    def test_plot_metric_over_time(self):
        """Should create plot without errors."""
        metric_values = [0.6, 0.7, 0.75, 0.78]
        visualise.plot_metric_over_time(metric_values, show=False)
        plt.close()
    
    def test_with_custom_labels(self):
        """Should accept custom labels."""
        metric_values = [0.6, 0.7, 0.75]
        visualise.plot_metric_over_time(metric_values, title="Accuracy",
                                       ylabel="Accuracy Score", show=False)
        plt.close()


class TestCompareModels:
    
    def test_compare_models(self):
        """Should compare two models."""
        metric1 = [0.60, 0.70, 0.75]
        metric2 = [0.65, 0.75, 0.80]
        visualise.compare_models(metric1, metric2, show=False)
        plt.close()
    
    def test_with_labels(self):
        """Should accept custom labels."""
        metric1 = [0.60, 0.70]
        metric2 = [0.65, 0.75]
        visualise.compare_models(metric1, metric2,
                               labels=("Tree", "Forest"), show=False)
        plt.close()


class TestPredictionsVsGroundTruth:
    
    def test_plot_predictions(self):
        """Should plot predictions vs ground truth."""
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        visualise.plot_predictions_vs_ground_truth(y_true, y_pred, show=False)
        plt.close()


class TestConfusionMatrix:
    
    def test_confusion_matrix(self):
        """Should plot confusion matrix."""
        cm = np.array([[80, 20], [10, 90]])
        visualise.plot_confusion_matrix(cm, show=False)
        plt.close()
    
    def test_with_labels(self):
        """Should accept class labels."""
        cm = np.array([[80, 20], [10, 90]])
        visualise.plot_confusion_matrix(cm, class_labels=['Class 0', 'Class 1'],
                                       show=False)
        plt.close()


class TestMultipleMetrics:
    
    def test_multiple_metrics(self):
        """Should plot multiple metrics."""
        metrics = {
            'Accuracy': [0.6, 0.7, 0.75],
            'Precision': [0.65, 0.72, 0.76]
        }
        visualise.plot_multiple_metrics(metrics, show=False)
        plt.close()


class TestFeatureImportance:
    
    def test_feature_importance(self):
        """Should plot feature importance."""
        importances = np.array([0.5, 0.3, 0.15, 0.05])
        visualise.plot_feature_importance(importances, show=False)
        plt.close()
    
    def test_with_names(self):
        """Should accept feature names."""
        importances = np.array([0.5, 0.3, 0.15])
        names = ['Age', 'Income', 'Score']
        visualise.plot_feature_importance(importances, feature_names=names, show=False)
        plt.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])