"""
stream.py - Streaming Training Manager

Orchestrates incremental model training, evaluation, and metric tracking.
Handles per-chunk updates with logging and history tracking.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import time


class StreamTrainer:
    """
    Manage streaming training of a model/pipeline.
    
    Coordinates:
    - Model/pipeline updates
    - Metric accumulation
    - Logging and statistics
    - History tracking
    
    Parameters:
        pipeline: Model or Pipeline to train
        metrics: Dict of {metric_name: metric_obj}
    
    Example:
        trainer = StreamTrainer(
            pipeline=pipe,
            metrics={
                'accuracy': StreamingAccuracy(),
                'precision': StreamingPrecision(n_classes=2)
            }
        )
        
        for X_chunk, y_chunk in chunks:
            results = trainer.fit_chunk(X_chunk, y_chunk)
            print(f"Accuracy: {results['accuracy']:.3f}")
    """
    
    def __init__(self, pipeline, metrics: Dict[str, object]):
        """
        Args:
            pipeline: Model or Pipeline with fit/partial_fit/predict/score
            metrics: Dict of {name: StreamingMetric} objects
        """
        self.pipeline = pipeline
        self.metrics = metrics
        
        # History tracking
        self.chunk_history = []
        self.metric_history = {name: [] for name in metrics.keys()}
        self.time_per_chunk = []
        self.total_samples = 0
        self.chunk_count = 0
    
    def fit_chunk(self, X_chunk: np.ndarray, y_chunk: np.ndarray, 
                  evaluate: bool = True) -> Dict[str, float]:
        """
        Train on a chunk and optionally evaluate.
        
        Flow:
        1. Time the training
        2. Call pipeline.partial_fit()
        3. Evaluate if requested
        4. Update metrics
        5. Log results
        
        Args:
            X_chunk: Feature chunk, shape (n_samples, n_features)
            y_chunk: Label chunk, shape (n_samples,)
            evaluate: Whether to evaluate on this chunk
        
        Returns:
            Dict of metric results: {metric_name: value, 'time': seconds}
        
        Example:
            results = trainer.fit_chunk(X_chunk, y_chunk)
            print(results)
            # {'accuracy': 0.75, 'precision': 0.80, 'time': 0.042}
        """
        start_time = time.time()
        
        # Train on chunk
        self.pipeline.partial_fit(X_chunk, y_chunk)
        
        chunk_time = time.time() - start_time
        self.time_per_chunk.append(chunk_time)
        
        results = {'time': chunk_time}
        
        if evaluate:
            # Evaluate on chunk
            y_pred = self.pipeline.predict(X_chunk)
            
            # Update all metrics
            for metric_name, metric in self.metrics.items():
                metric.update(y_chunk, y_pred)
                metric_value = metric.result()
                results[metric_name] = metric_value
                self.metric_history[metric_name].append(metric_value)
        
        # Update counters
        self.chunk_history.append(len(X_chunk))
        self.total_samples += len(X_chunk)
        self.chunk_count += 1
        
        return results
    
    def score_chunk(self, X_chunk: np.ndarray, y_chunk: np.ndarray) -> Dict[str, float]:
        """
        Evaluate on a chunk WITHOUT training.
        
        Useful for test sets or validation.
        
        Args:
            X_chunk: Feature chunk
            y_chunk: Label chunk
        
        Returns:
            Dict of metric results
        
        Example:
            test_results = trainer.score_chunk(X_test, y_test)
            print(f"Test Accuracy: {test_results['accuracy']:.3f}")
        """
        y_pred = self.pipeline.predict(X_chunk)
        
        results = {}
        for metric_name, metric in self.metrics.items():
            metric.update(y_chunk, y_pred)
            results[metric_name] = metric.result()
        
        return results
    
    def get_metric_history(self) -> Dict[str, List[float]]:
        """
        Get accumulated metrics over time.
        
        Returns:
            Dict of {metric_name: [value_chunk1, value_chunk2, ...]}
        
        Example:
            history = trainer.get_metric_history()
            accuracies = history['accuracy']  # [0.6, 0.7, 0.75, ...]
        """
        return self.metric_history
    
    def reset_metrics(self):
        """Reset all metrics for fresh evaluation."""
        for metric in self.metrics.values():
            metric.reset()
        
        self.metric_history = {name: [] for name in self.metrics.keys()}
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get overall training statistics.
        
        Returns statistics like:
        - Total samples processed
        - Number of chunks
        - Average time per chunk
        - Current metrics
        
        Example:
            stats = trainer.get_stats()
            print(f"Processed {stats['total_samples']} samples")
            print(f"Avg time/chunk: {stats['avg_time_per_chunk']:.3f}s")
        """
        avg_time = np.mean(self.time_per_chunk) if self.time_per_chunk else 0.0
        
        current_metrics = {
            name: metric.result() 
            for name, metric in self.metrics.items()
        }
        
        return {
            'total_samples': self.total_samples,
            'chunk_count': self.chunk_count,
            'avg_time_per_chunk': avg_time,
            'total_time': sum(self.time_per_chunk),
            'chunk_sizes': self.chunk_history,
            'current_metrics': current_metrics,
        }
    
    def print_summary(self):
        """Print training summary statistics."""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("STREAMING TRAINING SUMMARY")
        print("="*60)
        print(f"Total samples processed: {stats['total_samples']}")
        print(f"Number of chunks: {stats['chunk_count']}")
        print(f"Avg time per chunk: {stats['avg_time_per_chunk']:.3f}s")
        print(f"Total training time: {stats['total_time']:.3f}s")
        print("\nCurrent Metrics:")
        for metric_name, value in stats['current_metrics'].items():
            if isinstance(value, np.ndarray):
                print(f"  {metric_name}: (array shape {value.shape})")
            else:
                try:
                    print(f"  {metric_name}: {float(value):.4f}")
                except:
                    print(f"  {metric_name}: {value}")
        print("="*60 + "\n")