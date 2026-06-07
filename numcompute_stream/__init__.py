from .stats import StreamingMean, StreamingVariance
from .tree import DecisionTreeClassifier
from .ensemble import BaggingClassifier, RandomForestClassifier
from .metrics import (
    accuracy, precision_score, recall_score, f1_score,
    confusion_matrix, StreamingAccuracy, StreamingPrecision,
    StreamingRecall, StreamingF1, StreamingConfusionMatrix
)
from .preprocessing import StandardScaler, MinMaxScaler, SimpleImputer, OneHotEncoder
from .pipeline import Pipeline
from .stream import StreamTrainer
from . import visualise
from . import io

__version__ = "1.0.0"

__all__ = [
    'StreamingMean',
    'StreamingVariance',
    'DecisionTreeClassifier',
    'BaggingClassifier',
    'RandomForestClassifier',
    'accuracy',
    'precision_score',
    'recall_score',
    'f1_score',
    'confusion_matrix',
    'StreamingAccuracy',
    'StreamingPrecision',
    'StreamingRecall',
    'StreamingF1',
    'StreamingConfusionMatrix',
    'StandardScaler',
    'MinMaxScaler',
    'SimpleImputer',
    'OneHotEncoder',
    'Pipeline',
    'StreamTrainer',
    'visualise',
    'io',
]