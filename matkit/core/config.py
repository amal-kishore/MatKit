"""Configuration management for MatKit."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from dotenv import load_dotenv
from pydantic import Field, validator
try:
    from pydantic import BaseSettings
except ImportError:
    from pydantic_settings import BaseSettings

from matkit.core.exceptions import ConfigurationError


class Config(BaseSettings):
    """Global configuration for MatKit."""
    
    # API Keys
    mp_api_key: Optional[str] = Field(None, env="MP_API_KEY")
    
    # Paths
    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".matkit" / "cache",
        env="MATKIT_CACHE_DIR"
    )
    data_dir: Path = Field(
        default_factory=lambda: Path.home() / ".matkit" / "data",
        env="MATKIT_DATA_DIR"
    )
    
    # Performance
    n_jobs: int = Field(-1, env="MATKIT_N_JOBS")
    chunk_size: int = Field(1000, env="MATKIT_CHUNK_SIZE")
    use_gpu: bool = Field(True, env="MATKIT_USE_GPU")
    
    # Logging
    log_level: str = Field("INFO", env="MATKIT_LOG_LEVEL")
    log_file: Optional[Path] = Field(None, env="MATKIT_LOG_FILE")
    
    # Network
    request_timeout: float = Field(30.0, env="MATKIT_REQUEST_TIMEOUT")
    max_retries: int = Field(3, env="MATKIT_MAX_RETRIES")
    
    # Feature extraction
    feature_timeout: float = Field(300.0, env="MATKIT_FEATURE_TIMEOUT")
    skip_errors: bool = Field(True, env="MATKIT_SKIP_ERRORS")
    
    # Machine Learning
    random_state: int = Field(42, env="MATKIT_RANDOM_STATE")
    test_size: float = Field(0.2, env="MATKIT_TEST_SIZE")
    cv_folds: int = Field(5, env="MATKIT_CV_FOLDS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("n_jobs")
    def validate_n_jobs(cls, v: int) -> int:
        """Validate n_jobs parameter."""
        if v == -1:
            return os.cpu_count() or 1
        elif v < 1:
            raise ValueError("n_jobs must be -1 or a positive integer")
        return v
    
    @validator("cache_dir", "data_dir", "log_file", pre=True)
    def expand_path(cls, v: Optional[Union[str, Path]]) -> Optional[Path]:
        """Expand user paths."""
        if v is None:
            return None
        path = Path(v).expanduser().resolve()
        return path
    
    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}")
        return v.upper()
    
    def setup_directories(self) -> None:
        """Create necessary directories."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        # Load environment variables
        load_dotenv()
        _config = Config()
        _config.setup_directories()
    return _config


def set_config(config: Union[Config, Dict[str, Any]]) -> None:
    """
    Set the global configuration.
    
    Parameters
    ----------
    config : Config or dict
        Configuration object or dictionary
    """
    global _config
    if isinstance(config, dict):
        _config = Config(**config)
    elif isinstance(config, Config):
        _config = config
    else:
        raise ConfigurationError(
            f"Invalid configuration type: {type(config)}"
        )
    _config.setup_directories()


def reset_config() -> None:
    """Reset configuration to defaults."""
    global _config
    _config = None