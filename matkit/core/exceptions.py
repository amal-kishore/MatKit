"""Custom exceptions for MatKit."""

from typing import Any, Optional


class MatKitError(Exception):
    """Base exception for all MatKit errors."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DataError(MatKitError):
    """Raised when data-related errors occur."""
    pass


class FeaturizationError(MatKitError):
    """Raised when featurization fails."""
    pass


class ValidationError(MatKitError):
    """Raised when validation fails."""
    pass


class ConfigurationError(MatKitError):
    """Raised when configuration is invalid."""
    pass


class APIError(MatKitError):
    """Raised when external API calls fail."""
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        details: Optional[dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.status_code = status_code
        self.response_body = response_body


class ModelError(MatKitError):
    """Raised when model-related errors occur."""
    pass


class VisualizationError(MatKitError):
    """Raised when visualization fails."""
    pass