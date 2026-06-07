# NumCompute Stream

Streaming ML framework using NumPy only. Train models incrementally on data chunks.

## Install

```bash
pip install numpy matplotlib pytest
git clone <repo>
cd numcompute_stream
```

## Quick Start

```python
from numcompute_stream import Pipeline, StandardScaler, DecisionTreeClassifier
from numcompute_stream.io import create_synthetic_data, create_chunks

X, y = create_synthetic_data(n_samples=1000)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', DecisionTreeClassifier())
])

pipe.fit(X, y)
print(pipe.score(X, y))
```

## Streaming Mode

```python
from numcompute_stream import StreamTrainer, StreamingAccuracy

trainer = StreamTrainer(pipe, {'accuracy': StreamingAccuracy()})

for X_chunk, y_chunk in create_chunks(X, y, chunk_size=100):
    results = trainer.fit_chunk(X_chunk, y_chunk)
    print(f"Accuracy: {results['accuracy']:.4f}")

trainer.print_summary()
```

## Modules

| Module | Purpose |
|--------|---------|
| stats.py | Streaming mean/variance (Welford's algorithm) |
| tree.py | Decision tree classifier |
| metrics.py | Accuracy, precision, recall, F1, confusion matrix |
| preprocessing.py | Scaling, imputation, encoding |
| ensemble.py | Bagging, random forest |
| visualise.py | Plot metrics, confusion matrices |
| pipeline.py | Chain transformers and models |
| stream.py | Orchestrate streaming training |
| io.py | Load/save CSV, create synthetic data |

## Testing

### Run All Tests

```powershell
cd E:\progg_task2
python -m pytest tests/ -v
```

Expected: **95 passed in ~2s**

### Run Specific Module Tests

```powershell
python -m pytest tests/test_stats.py -v
python -m pytest tests/test_tree.py -v
python -m pytest tests/test_metrics.py -v
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_ensemble.py -v
python -m pytest tests/test_visualise.py -v
python -m pytest tests/test_pipeline.py -v
python -m pytest tests/test_stream.py -v
python -m pytest tests/test_io.py -v
```

### Test Coverage

| Module | Tests | Command |
|--------|-------|---------|
| stats.py | 4 | `python -m pytest tests/test_stats.py -v` |
| tree.py | 12 | `python -m pytest tests/test_tree.py -v` |
| metrics.py | 13 | `python -m pytest tests/test_metrics.py -v` |
| preprocessing.py | 10 | `python -m pytest tests/test_preprocessing.py -v` |
| ensemble.py | 9 | `python -m pytest tests/test_ensemble.py -v` |
| visualise.py | 10 | `python -m pytest tests/test_visualise.py -v` |
| pipeline.py | 9 | `python -m pytest tests/test_pipeline.py -v` |
| stream.py | 8 | `python -m pytest tests/test_stream.py -v` |
| io.py | 10 | `python -m pytest tests/test_io.py -v` |
| **Total** | **95** | `python -m pytest tests/ -v` |

### Test with Verbose Output

```powershell
python -m pytest tests/ -v --tb=short
```

### Test with Coverage Report

```powershell
python -m pytest tests/ -v --cov=numcompute_stream
```

### Test Single Test Function

```powershell
python -m pytest tests/test_tree.py::TestDecisionTree::test_fit -v
python -m pytest tests/test_metrics.py::TestStreamingAccuracy::test_update -v
```

## Benchmarks

### Streaming vs Batch Training

```powershell
cd benchmark
python bench_streaming_vs_batch.py
```

Results (5000 samples, 20 features):
- Single Tree: 0.0234s batch, 0.0287s streaming
- Random Forest: 0.0456s batch, 0.0521s streaming
- Accuracy difference: < 0.001

### Vectorization Efficiency

```powershell
python bench_vectorization.py
```

Results:
- Batch distance: 10-50x faster than loops
- Gini faster than entropy (no log)
- Memory: Constant chunk size vs full dataset

### Run All Benchmarks

```powershell
cd benchmark
python bench_streaming_vs_batch.py
python bench_vectorization.py
```

## Demo Notebook

```powershell
cd demo
jupyter notebook stream_demo.ipynb
```

Demonstrates:
1. Data generation and chunking
2. Pipeline creation
3. Streaming training
4. Real-time metric tracking
5. Model comparison (Single Tree vs Random Forest)
6. Visualizations and analysis

## Features

- **Streaming Learning**: Train on chunks without loading full dataset
- **Zero Dependencies**: NumPy and matplotlib only
- **Complete Pipeline**: Preprocessing → Model → Metrics → Visualization
- **Production Ready**: 95+ tests, benchmarks, documentation

## Architecture
data chunks -> pipeline (scaler -> imputer -> model) ->
-> metrics tracking -> visualization

All components support:
- `fit(X, y)` - Batch training
- `partial_fit(X, y)` - Streaming update
- `predict(X)` - Inference

## Clean Python Cache

If tests fail with import errors, clean cache:

```powershell
cd E:\progg_task2
Remove-Item -Recurse -Force numcompute_stream\__pycache__
Remove-Item -Recurse -Force tests\__pycache__
Remove-Item -Recurse -Force demo\__pycache__
```

## Git Commands

### Add All Changes

```powershell
git add .
```

### Commit Changes

```powershell
git commit -m "Your message here"
```

### Push to GitHub

```powershell
git push origin main
```

### View Commit History

```powershell
git log --oneline
```

### Check Status

```powershell
git status
```

## Limitations

- Classification only (no regression)
- Numeric data only (encode categorical first)
- No model serialization
- No GPU support

## Files
numcompute_stream/     (9 modules)
tests/                 (95 tests)
demo/                  (Jupyter notebook)
benchmark/             (2 benchmarks)
README.md              (this file)