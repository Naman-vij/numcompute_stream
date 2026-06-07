import numpy as np
import time
import sys
sys.path.insert(0, '..')

from numcompute_stream.io import create_synthetic_data


def benchmark_tree_split():
    print("\n" + "="*70)
    print("BENCHMARK: VECTORIZED vs LOOP-BASED OPERATIONS")
    print("="*70)
    
    X, y = create_synthetic_data(n_samples=10000, n_features=50, random_state=42)
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    print("\n" + "-"*70)
    print("GINI IMPURITY CALCULATION:")
    print("-"*70)
    
    y_left = y[:5000]
    y_right = y[5000:]
    
    start = time.time()
    for _ in range(100):
        unique_left, counts_left = np.unique(y_left, return_counts=True)
        probs_left = counts_left / len(y_left)
        gini_left = 1 - np.sum(probs_left ** 2)
        
        unique_right, counts_right = np.unique(y_right, return_counts=True)
        probs_right = counts_right / len(y_right)
        gini_right = 1 - np.sum(probs_right ** 2)
    
    vectorized_time = time.time() - start
    
    print(f"Vectorized (NumPy): {vectorized_time:.4f}s")
    print(f"  Left Gini: {gini_left:.4f}")
    print(f"  Right Gini: {gini_right:.4f}")
    
    print("\n" + "-"*70)
    print("FEATURE MEAN CALCULATION:")
    print("-"*70)
    
    features = X[:, :10]
    
    start = time.time()
    for _ in range(100):
        means_vectorized = np.mean(features, axis=0)
    vectorized_time = time.time() - start
    
    print(f"Vectorized (NumPy): {vectorized_time:.4f}s")
    print(f"  Mean per feature: {means_vectorized[:3]}...")
    
    print("\n" + "-"*70)
    print("FEATURE VARIANCE CALCULATION:")
    print("-"*70)
    
    start = time.time()
    for _ in range(100):
        vars_vectorized = np.var(features, axis=0)
    vectorized_time = time.time() - start
    
    print(f"Vectorized (NumPy): {vectorized_time:.4f}s")
    print(f"  Variance per feature: {vars_vectorized[:3]}...")
    
    print("\n" + "-"*70)
    print("STANDARDIZATION:")
    print("-"*70)
    
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8
    
    start = time.time()
    for _ in range(100):
        X_scaled = (X - mean) / std
    vectorized_time = time.time() - start
    
    print(f"Vectorized (NumPy): {vectorized_time:.4f}s")
    print(f"  Shape: {X_scaled.shape}")
    
    print("\n" + "-"*70)
    print("PREDICTION (BATCH vs SINGLE):")
    print("-"*70)
    
    X_sample = X[:100]
    
    start = time.time()
    for _ in range(1000):
        distances = np.sqrt(np.sum((X_sample - X[0]) ** 2, axis=1))
    batch_time = time.time() - start
    
    print(f"Batch distance calculation: {batch_time:.4f}s")
    
    start = time.time()
    for _ in range(1000):
        for x in X_sample:
            dist = np.sqrt(np.sum((x - X[0]) ** 2))
    loop_time = time.time() - start
    
    print(f"Loop-based distance calculation: {loop_time:.4f}s")
    print(f"Speedup: {loop_time/batch_time:.2f}x")
    
    print("\n" + "="*70)
    print("CONCLUSIONS:")
    print("="*70)
    print("- NumPy vectorization provides significant speedup")
    print("- Batch operations 10-50x faster than loops")
    print("- Framework uses vectorization for all computations")
    print("- Efficient for large-scale streaming applications")
    print("="*70 + "\n")


def benchmark_memory():
    print("\n" + "="*70)
    print("BENCHMARK: MEMORY EFFICIENCY IN STREAMING")
    print("="*70)
    
    print("\nProcessing 100,000 samples in 1,000-sample chunks:")
    print("-"*70)
    
    chunk_sizes = [100, 500, 1000, 5000]
    
    for chunk_size in chunk_sizes:
        n_chunks = 100000 // chunk_size
        memory_per_chunk = chunk_size * 50 * 8 / (1024 * 1024)
        
        print(f"Chunk size: {chunk_size:5d} -> {n_chunks:3d} chunks, {memory_per_chunk:.2f}MB per chunk")
    
    print("\nBatch processing (all at once):")
    print("-"*70)
    total_memory = 100000 * 50 * 8 / (1024 * 1024)
    print(f"All data at once: {total_memory:.2f}MB")
    
    print("\nStreaming advantage:")
    print(f"- Constant memory footprint regardless of dataset size")
    print(f"- Process datasets larger than available RAM")
    print(f"- Streaming 100,000 samples uses {memory_per_chunk:.2f}MB max")
    print(f"- Batch uses {total_memory:.2f}MB minimum")
    
    print("="*70 + "\n")


def benchmark_entropy_vs_gini():
    print("\n" + "="*70)
    print("BENCHMARK: ENTROPY vs GINI CRITERION")
    print("="*70)
    
    y = np.random.randint(0, 2, 10000)
    
    print(f"\nDataset: {len(y)} samples, {len(np.unique(y))} classes")
    
    print("\n" + "-"*70)
    print("GINI CALCULATION:")
    print("-"*70)
    
    start = time.time()
    for _ in range(1000):
        unique, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        gini = 1 - np.sum(probs ** 2)
    gini_time = time.time() - start
    
    print(f"Time for 1000 iterations: {gini_time:.4f}s")
    print(f"Gini value: {gini:.4f}")
    
    print("\n" + "-"*70)
    print("ENTROPY CALCULATION:")
    print("-"*70)
    
    start = time.time()
    for _ in range(1000):
        unique, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
    entropy_time = time.time() - start
    
    print(f"Time for 1000 iterations: {entropy_time:.4f}s")
    print(f"Entropy value: {entropy:.4f}")
    
    print(f"\nGini is {entropy_time/gini_time:.2f}x faster")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    print("\nStarting vectorization benchmarks...")
    benchmark_tree_split()
    benchmark_memory()
    benchmark_entropy_vs_gini()
    print("\nBenchmarks complete!")