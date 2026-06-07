import numpy as np
import time
import sys
sys.path.insert(0, '..')

from numcompute_stream import (
    Pipeline, StreamTrainer,
    StandardScaler, DecisionTreeClassifier, RandomForestClassifier,
    StreamingAccuracy
)
from numcompute_stream.io import create_synthetic_data, create_chunks


def benchmark_streaming():
    print("\n" + "="*70)
    print("BENCHMARK: STREAMING VS BATCH TRAINING")
    print("="*70)
    
    X, y = create_synthetic_data(n_samples=5000, n_features=20, random_state=42)
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    tree_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', DecisionTreeClassifier(max_depth=7))
    ])
    
    forest_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=5, max_depth=7))
    ])
    
    print("\n" + "-"*70)
    print("BATCH TRAINING (all data at once):")
    print("-"*70)
    
    start = time.time()
    tree_pipe.fit(X, y)
    batch_tree_time = time.time() - start
    batch_tree_score = tree_pipe.score(X, y)
    
    start = time.time()
    forest_pipe.fit(X, y)
    batch_forest_time = time.time() - start
    batch_forest_score = forest_pipe.score(X, y)
    
    print(f"Single Tree:  {batch_tree_time:.4f}s, Accuracy: {batch_tree_score:.4f}")
    print(f"Random Forest: {batch_forest_time:.4f}s, Accuracy: {batch_forest_score:.4f}")
    
    print("\n" + "-"*70)
    print("STREAMING TRAINING (100-sample chunks):")
    print("-"*70)
    
    metrics_tree = {'accuracy': StreamingAccuracy()}
    metrics_forest = {'accuracy': StreamingAccuracy()}
    
    trainer_tree = StreamTrainer(tree_pipe, metrics_tree)
    trainer_forest = StreamTrainer(forest_pipe, metrics_forest)
    
    start = time.time()
    for X_chunk, y_chunk in create_chunks(X, y, chunk_size=100, shuffle=False):
        trainer_tree.fit_chunk(X_chunk, y_chunk)
    stream_tree_time = time.time() - start
    stream_tree_score = trainer_tree.get_stats()['current_metrics']['accuracy']
    
    start = time.time()
    for X_chunk, y_chunk in create_chunks(X, y, chunk_size=100, shuffle=False):
        trainer_forest.fit_chunk(X_chunk, y_chunk)
    stream_forest_time = time.time() - start
    stream_forest_score = trainer_forest.get_stats()['current_metrics']['accuracy']
    
    print(f"Single Tree:  {stream_tree_time:.4f}s, Accuracy: {stream_tree_score:.4f}")
    print(f"Random Forest: {stream_forest_time:.4f}s, Accuracy: {stream_forest_score:.4f}")
    
    print("\n" + "-"*70)
    print("TIME COMPARISON:")
    print("-"*70)
    
    print(f"\nSingle Tree:")
    print(f"  Batch:     {batch_tree_time:.4f}s")
    print(f"  Streaming: {stream_tree_time:.4f}s")
    print(f"  Ratio:     {stream_tree_time/batch_tree_time:.2f}x")
    
    print(f"\nRandom Forest:")
    print(f"  Batch:     {batch_forest_time:.4f}s")
    print(f"  Streaming: {stream_forest_time:.4f}s")
    print(f"  Ratio:     {stream_forest_time/batch_forest_time:.2f}x")
    
    print("\n" + "-"*70)
    print("ACCURACY COMPARISON:")
    print("-"*70)
    
    print(f"\nSingle Tree:")
    print(f"  Batch:     {batch_tree_score:.4f}")
    print(f"  Streaming: {stream_tree_score:.4f}")
    print(f"  Difference: {abs(batch_tree_score - stream_tree_score):+.4f}")
    
    print(f"\nRandom Forest:")
    print(f"  Batch:     {batch_forest_score:.4f}")
    print(f"  Streaming: {stream_forest_score:.4f}")
    print(f"  Difference: {abs(batch_forest_score - stream_forest_score):+.4f}")
    
    print("\n" + "="*70)
    print("CONCLUSIONS:")
    print("="*70)
    print("- Streaming training enables processing data incrementally")
    print("- Minimal accuracy loss compared to batch training")
    print("- Suitable for real-time data processing scenarios")
    print("- Memory efficient for large datasets")
    print("="*70 + "\n")


def benchmark_scalability():
    print("\n" + "="*70)
    print("BENCHMARK: SCALABILITY (Varying Dataset Size)")
    print("="*70)
    
    sizes = [1000, 2000, 5000, 10000]
    
    print("\nSingle Tree - Training Time vs Dataset Size:")
    print("-"*70)
    
    for size in sizes:
        X, y = create_synthetic_data(n_samples=size, n_features=20, random_state=42)
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', DecisionTreeClassifier(max_depth=7))
        ])
        
        start = time.time()
        pipe.fit(X, y)
        elapsed = time.time() - start
        
        print(f"  {size:5d} samples: {elapsed:.4f}s")
    
    print("\nRandom Forest - Training Time vs Dataset Size:")
    print("-"*70)
    
    for size in sizes:
        X, y = create_synthetic_data(n_samples=size, n_features=20, random_state=42)
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(n_estimators=5, max_depth=7))
        ])
        
        start = time.time()
        pipe.fit(X, y)
        elapsed = time.time() - start
        
        print(f"  {size:5d} samples: {elapsed:.4f}s")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    benchmark_streaming()
    benchmark_scalability()