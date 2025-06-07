"""Base classes for MatKit components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pydantic import BaseModel, Field

from matkit.core.config import get_config
from matkit.core.exceptions import FeaturizationError, ValidationError
from matkit.core.logging import get_logger

logger = get_logger(__name__)


class Citation(BaseModel):
    """Citation information for methods."""
    
    authors: List[str]
    title: str
    journal: str
    year: int
    doi: Optional[str] = None
    url: Optional[str] = None
    
    def __str__(self) -> str:
        """Format citation as string."""
        citation = f"{', '.join(self.authors)} ({self.year}). "
        citation += f"{self.title}. {self.journal}."
        if self.doi:
            citation += f" DOI: {self.doi}"
        return citation


class BaseFeaturizer(ABC):
    """
    Base class for all featurizers.
    
    This class provides the interface and common functionality for all
    featurizers in MatKit. It handles parallel processing, error handling,
    and progress tracking.
    """
    
    def __init__(
        self,
        n_jobs: Optional[int] = None,
        skip_errors: Optional[bool] = None,
        return_errors: bool = False,
        verbose: bool = True
    ):
        """
        Initialize BaseFeaturizer.
        
        Parameters
        ----------
        n_jobs : int, optional
            Number of parallel jobs. If None, uses config default
        skip_errors : bool, optional
            Whether to skip errors during featurization
        return_errors : bool, default=False
            Whether to return error information
        verbose : bool, default=True
            Whether to show progress information
        """
        config = get_config()
        self.n_jobs = n_jobs or config.n_jobs
        self.skip_errors = skip_errors if skip_errors is not None else config.skip_errors
        self.return_errors = return_errors
        self.verbose = verbose
        self._feature_labels: Optional[List[str]] = None
        self._fitted = False
    
    @abstractmethod
    def featurize(self, data: Any) -> np.ndarray:
        """
        Generate features for a single data point.
        
        Parameters
        ----------
        data : Any
            Input data (composition, structure, etc.)
            
        Returns
        -------
        np.ndarray
            Feature vector
        """
        pass
    
    @abstractmethod
    def feature_labels(self) -> List[str]:
        """
        Get feature labels.
        
        Returns
        -------
        List[str]
            List of feature names
        """
        pass
    
    @abstractmethod
    def citations(self) -> List[Citation]:
        """
        Get citations for this featurizer.
        
        Returns
        -------
        List[Citation]
            List of relevant citations
        """
        pass
    
    def fit(self, X: Union[pd.DataFrame, List[Any]], y: Optional[Any] = None) -> "BaseFeaturizer":
        """
        Fit the featurizer to data.
        
        Parameters
        ----------
        X : DataFrame or list
            Input data
        y : array-like, optional
            Target values (ignored)
            
        Returns
        -------
        self
            Fitted featurizer
        """
        self._fitted = True
        return self
    
    def transform(self, X: Union[pd.DataFrame, List[Any]]) -> pd.DataFrame:
        """
        Transform data into features.
        
        Parameters
        ----------
        X : DataFrame or list
            Input data
            
        Returns
        -------
        DataFrame
            Feature matrix
        """
        if not self._fitted:
            raise ValidationError("Featurizer must be fitted before transform")
        
        # Convert to list if needed
        if isinstance(X, pd.DataFrame):
            data_list = X.values.tolist() if X.shape[1] == 1 else X.to_dict("records")
        else:
            data_list = X
        
        # Process in parallel
        if self.n_jobs == 1:
            results = [self._featurize_wrapper(x) for x in data_list]
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._featurize_wrapper)(x) for x in data_list
            )
        
        # Separate features and errors
        features = []
        errors = []
        
        for result in results:
            if isinstance(result, Exception):
                if not self.skip_errors:
                    raise result
                features.append(np.full(len(self.feature_labels()), np.nan))
                errors.append(str(result))
            else:
                features.append(result)
                errors.append(None)
        
        # Create DataFrame
        df = pd.DataFrame(features, columns=self.feature_labels())
        
        if self.return_errors:
            df["_errors"] = errors
        
        return df
    
    def fit_transform(self, X: Union[pd.DataFrame, List[Any]], y: Optional[Any] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _featurize_wrapper(self, data: Any) -> Union[np.ndarray, Exception]:
        """Wrapper for featurize with error handling."""
        try:
            return self.featurize(data)
        except Exception as e:
            logger.warning(f"Featurization failed: {e}")
            return e
    
    def __repr__(self) -> str:
        """String representation."""
        params = []
        if self.n_jobs != -1:
            params.append(f"n_jobs={self.n_jobs}")
        if not self.skip_errors:
            params.append("skip_errors=False")
        if self.return_errors:
            params.append("return_errors=True")
        
        param_str = ", ".join(params)
        return f"{self.__class__.__name__}({param_str})"


class BaseTransformer(ABC):
    """Base class for data transformers."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BaseTransformer":
        """Fit the transformer."""
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        pass
    
    @abstractmethod
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform the data."""
        pass
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class BaseModel(ABC):
    """Base class for machine learning models."""
    
    def __init__(self, random_state: Optional[int] = None):
        """Initialize model."""
        config = get_config()
        self.random_state = random_state or config.random_state
        self._fitted = False
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Fit the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Score the model."""
        pass
    
    def fit_predict(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Fit and predict in one step."""
        return self.fit(X, y).predict(X)