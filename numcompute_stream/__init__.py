"""
NumCompute Stream - Streaming Machine Learning Framework

A streaming-compatible decision tree and ensemble learning library
built with NumPy and Matplotlib.
"""

from .stats import StreamingMean, StreamingVariance

# TODO: Uncomment as you complete each module
# from .tree import DecisionTreeClassifier
# from .ensemble import BaggingClassifier, RandomForestClassifier
# from .preprocessing import StandardScaler, MinMaxScaler, SimpleImputer, OneHotEncoder
# from .metrics import accuracy, precision_score, recall_score, f1_score
# from .pipeline import Pipeline
# from .stream import StreamTrainer
# from . import visualise

# __version__ = "1.0.0"

__all__ = [
    'StreamingMean',
    'StreamingVariance',
]