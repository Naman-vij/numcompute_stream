"""
pipeline.py - data preprocessing pipeline

chains preprocessing steps and a final estimator.
supports both batch (fit) and streaming (partial_fit) modes.
"""

import numpy as np
from typing import List, Optional, Tuple, Optional

class Pipeline:
    """
    chain preprocessing steps and a final estimator.

    How it works:
    1. transformers process data step by step
    2. final estimator makes predictions
    3. fit() trains all components
    4. partial_fit() updates all components (streaming)
    5. predict() applies transformations then predicts 
       Example:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('imputer', SimpleImputer()),
            ('model', DecisionTreeClassifier())
        ])
        pipe.fit(X, y)
        y_pred = pipe.predict(X_new)
    
    Parameters:
        steps: List of (name, transformer) tuples
               Last element should be ('model', estimator)
             """
    def __init__(self, steps: List[Tuple[str, object]]):
        """
        Args:
            steps: List of (name, transformer) tuples.
                   All but last must have fit/transform/partial_fit.
                   Last must be an estimator with fit/predict/partial_fit.
        """
        if not steps:
            raise ValueError("Pipeline atleast have one step")
        
        self.steps = steps
        self.transformers = steps[:-1]
        self.model = steps[-1][1]

        # validate last step is an estimator
        if not hasattr(self.model, 'predict'):
            raise ValueError("Last step must be an estimator with predict() method")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Pipeline':
        """
        fit all transformers and the model

        Flow:
        Flow:
        1. X → Transformer 1 → X1
        2. X1 → Transformer 2 → X2
        3. X2 → ... → Xn
        4. Xn → Model.fit(y)
        
        Args:
            X: Features, shape (n_samples, n_features)
            y: Labels, shape (n_samples,)
        
        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # fit and transoform through pipleline
        for name, transformer in self.transformers:
            X = transformer.fit(X).transform(X)

        # fit final model
        self.model.fit(X,y)

        return self
    
    def partial_fit(self, X: np.ndarray, y:np.ndarray) -> 'Pipeline':
        """
        Update all transformers and model with new chunk (streaming).
        
        Flow (same as fit, but uses partial_fit):
        1. X → Transformer 1.partial_fit → X1
        2. X1 → Transformer 2.partial_fit → X2
        3. X2 → ... → Xn
        4. Xn → Model.partial_fit(y)
        
        Args:
            X: Feature chunk, shape (n_samples, n_features)
            y: Label chunk, shape (n_samples,)
        
        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # update transformers
        for name, transformer in self.transformers:
            if hasattr(transformer, 'partial_fit'):
                transformer.partial_fit(X)
            X = transformer.transform(X)

        # update model
        for name, transformer in self.transformers:
            if hasattr(transformer, 'partial_fit'):
                transformer.partial_fit(X)
            X = transformer.transform(X)

        # update transformers
        for name, transformer in self.transformers:
            if hasattr(transformer, 'partial_fit'):
                transformer.partial_fit(X)
            X = transformer.transform(X)

        # update model
        self.model.partial_fit(X,y)

        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data and predict.
        
        Flow:
        1. X → Transformer 1 → X1
        2. X1 → Transformer 2 → X2
        3. X2 → ... → Xn
        4. Xn → Model.predict() → y_pred
        
        Args:
            X: Features, shape (n_samples, n_features)
        
        Returns:
            Predictions, shape (n_samples,)
        """
        X = np.asarray(X)

        # Transform through pipeline
        for name, transformer in self.transformers:
            X = transformer.transform(X)

        # predict
        return self.model.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Transform and score on test set.
        
        Args:
            X: Features
            y: True labels
        
        Returns:
            Accuracy score (0 to 1)
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def get_params(self, deep: bool = True) -> dict:
        """Get pipeline parameters."""
        params = {}
        for name, step in self.steps:
            if hasattr(step, 'get_params'):
                step_params = step.get_params(deep=deep)
                for key, val in step_params.items():
                    params[f"{name}__{key}"] = val
            else:
                params[name] = step
            return params
        
    def set_params(self, **params) -> 'Pipeline':
        """Set pipeline parameters"""
        for name, step in self.steps:
            step_params = {}
            for key, val in params.items():
                if key.startswith(f"{name}__"):
                    clean_key = key.replace(f"{name}__", "")
                    step_params[clean_key] = val

            if step_params and hasattr(step, 'set_params'):
                step.set_params(**step_params)

        return self
