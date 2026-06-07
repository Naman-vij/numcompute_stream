"""
visualise.py - Visualization Module

Provides matplotlib-based plotting functions for metrics, predictions, and model comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


def plot_metric_over_time(metric_values: List[float], title: str = "Metric Over Time",
                          ylabel: str = "Metric Value", xlabel: str = "Chunk",
                          save_path: Optional[str] = None, show: bool = True):
    """
    Plot a single metric across chunks.
    
    Shows how metric changes as more data arrives.
    
    Args:
        metric_values: List of metric values (one per chunk)
        title: Plot title
        ylabel: Y-axis label
        xlabel: X-axis label
        save_path: If provided, save to this path
        show: Whether to display plot
    
    Example:
        accuracies = [0.60, 0.70, 0.75, 0.78]
        plot_metric_over_time(accuracies, title="Model Accuracy Over Time")
    """
    plt.figure(figsize=(10, 6))
    
    chunks = np.arange(len(metric_values))
    plt.plot(chunks, metric_values, marker='o', linewidth=2, markersize=6, label='Metric')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def compare_models(metric1: List[float], metric2: List[float],
                   labels: Tuple[str, str] = ("Model 1", "Model 2"),
                   title: str = "Model Comparison", ylabel: str = "Accuracy",
                   xlabel: str = "Chunk", save_path: Optional[str] = None,
                   show: bool = True):
    """
    Compare two models' metrics over time.
    
    Shows how two models perform side-by-side.
    Useful for: single tree vs ensemble, different hyperparameters, etc.
    
    Args:
        metric1: Metric values for model 1
        metric2: Metric values for model 2
        labels: Tuple of (label1, label2) for legend
        title: Plot title
        ylabel: Y-axis label
        xlabel: X-axis label
        save_path: If provided, save to this path
        show: Whether to display plot
    
    Example:
        tree_acc = [0.60, 0.70, 0.75]
        forest_acc = [0.65, 0.75, 0.80]
        compare_models(tree_acc, forest_acc,
                      labels=("Single Tree", "Random Forest"))
    """
    plt.figure(figsize=(12, 6))
    
    chunks = np.arange(len(metric1))
    plt.plot(chunks, metric1, marker='o', linewidth=2, markersize=6, label=labels[0])
    plt.plot(chunks, metric2, marker='s', linewidth=2, markersize=6, label=labels[1])
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_predictions_vs_ground_truth(y_true: np.ndarray, y_pred: np.ndarray,
                                     title: str = "Predictions vs Ground Truth",
                                     save_path: Optional[str] = None,
                                     show: bool = True):
    """
    Visualize predictions vs actual labels.
    
    Shows if model predicts different class distribution than actual.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        save_path: If provided, save to this path
        show: Whether to display plot
    
    Example:
        y_true = [0, 1, 1, 0, 1, 1]
        y_pred = [0, 1, 0, 0, 1, 1]
        plot_predictions_vs_ground_truth(y_true, y_pred)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # True labels distribution
    unique_true = np.unique(y_true)
    counts_true = np.array([np.sum(y_true == c) for c in unique_true])
    ax1.bar(unique_true, counts_true, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel("Class", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Ground Truth Distribution", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Predicted labels distribution
    unique_pred = np.unique(y_pred)
    counts_pred = np.array([np.sum(y_pred == c) for c in unique_pred])
    ax2.bar(unique_pred, counts_pred, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel("Class", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Predicted Distribution", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(confusion_matrix: np.ndarray, class_labels: Optional[List[str]] = None,
                          title: str = "Confusion Matrix", save_path: Optional[str] = None,
                          show: bool = True):
    """
    Plot confusion matrix as heatmap.
    
    Shows:
    - Diagonal (top-left to bottom-right) = correct predictions
    - Off-diagonal = mistakes
    
    Args:
        confusion_matrix: Confusion matrix, shape (n_classes, n_classes)
        class_labels: Optional labels for classes
        title: Plot title
        save_path: If provided, save to this path
        show: Whether to display plot
    
    Example:
        cm = np.array([[80, 20], [10, 90]])
        plot_confusion_matrix(cm, class_labels=['Class 0', 'Class 1'])
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    n_classes = confusion_matrix.shape[0]
    if class_labels is None:
        class_labels = [str(i) for i in range(n_classes)]
    
    # Normalize for display (percentages)
    cm_normalized = confusion_matrix.astype(float) / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-10)
    
    im = ax.imshow(cm_normalized, cmap='Blues', aspect='auto')
    
    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            count = confusion_matrix[i, j]
            percent = cm_normalized[i, j]
            text = ax.text(j, i, f'{count}\n({percent:.1%})',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label="Normalized Count")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_multiple_metrics(metrics_dict: dict, title: str = "Multiple Metrics",
                         xlabel: str = "Chunk", save_path: Optional[str] = None,
                         show: bool = True):
    """
    Plot multiple metrics on same plot.
    
    Useful for: comparing accuracy, precision, recall, F1 on same chart.
    
    Args:
        metrics_dict: Dict of {metric_name: values_list}
        title: Plot title
        xlabel: X-axis label
        save_path: If provided, save to this path
        show: Whether to display plot
    
    Example:
        metrics = {
            'Accuracy': [0.60, 0.70, 0.75],
            'Precision': [0.65, 0.72, 0.76],
            'Recall': [0.58, 0.68, 0.74]
        }
        plot_multiple_metrics(metrics, title="Model Performance Metrics")
    """
    plt.figure(figsize=(12, 6))
    
    for metric_name, values in metrics_dict.items():
        chunks = np.arange(len(values))
        plt.plot(chunks, values, marker='o', linewidth=2, label=metric_name)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_feature_importance(importances: np.ndarray, feature_names: Optional[List[str]] = None,
                           top_k: int = 10, title: str = "Feature Importance",
                           save_path: Optional[str] = None, show: bool = True):
    """
    Plot top-k important features.
    
    Shows which features the model relies on most.
    
    Args:
        importances: Feature importance scores
        feature_names: Optional feature names
        top_k: Number of top features to display
        title: Plot title
        save_path: If provided, save to this path
        show: Whether to display plot
    
    Example:
        importances = np.array([0.5, 0.3, 0.15, 0.05])
        feature_names = ['Age', 'Income', 'Credit Score', 'Employment']
        plot_feature_importance(importances, feature_names, top_k=3)
    """
    importances = np.asarray(importances)
    
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]
    
    # Sort and get top-k
    sorted_idx = np.argsort(importances)[::-1][:top_k]
    top_importances = importances[sorted_idx]
    top_names = [feature_names[i] for i in sorted_idx]
    
    plt.figure(figsize=(10, 6))
    plt.barh(top_names, top_importances, color='steelblue', alpha=0.8, edgecolor='black')
    plt.xlabel("Importance", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()