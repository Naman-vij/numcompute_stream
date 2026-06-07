"""
NumCompute Stream - Streaming Machine Learning Framework

A streaming-compatible decision tree and ensemble learning library
built with NumPy and Matplotlib.
"""

from .stats import StreamingMean, StreamingVariance
from .tree import DecisionTreeClassifier
# from .ensemble import BaggingClassifier, RandomForestClassifier
from .preprocessing import StandardScaler, MinMaxScaler, SimpleImputer, OneHotEncoder
from .metrics import accuracy, precision_score, recall_score, f1_score
# from .pipeline import Pipeline
# from .stream import StreamTrainer
# from . import visualise


__all__ = [
    'StreamingMean',
    'StreamingVariance',
    'DecisionTreeClassifier',
    'accuracy',
    'precision_score',
    'recall_score',
    'f1_score',
    'StandardScaler',
    'MinMaxScaler',
    'SimpleImputer',
    'OneHotEncoder',
]