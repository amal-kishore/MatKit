"""Caching system for MatKit data operations."""

import hashlib
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from matkit.core.config import get_config
from matkit.core.logging import get_logger

logger = get_logger(__name__)


class DataCache:
    """
    Persistent cache for materials data.
    
    Features:
    - Automatic serialization/deserialization
    - TTL (time-to-live) support
    - Size limits
    - Multiple backend support
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        ttl: Optional[timedelta] = None,
        max_size_mb: float = 1000.0,
        backend: str = "disk"
    ):
        """
        Initialize cache.
        
        Parameters
        ----------
        cache_dir : Path, optional
            Cache directory. If None, uses config default
        ttl : timedelta, optional
            Time-to-live for cache entries
        max_size_mb : float, default=1000.0
            Maximum cache size in MB
        backend : str, default="disk"
            Cache backend ("disk" or "memory")
        """
        config = get_config()
        self.cache_dir = cache_dir or config.cache_dir
        self.ttl = ttl or timedelta(days=7)
        self.max_size_mb = max_size_mb
        self.backend = backend
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory cache for fast access
        self._memory_cache = {} if backend == "memory" else None
        
        # Metadata file
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def _get_cache_key(self, key: str) -> str:
        """Generate cache key hash."""
        return hashlib.sha256(str(key).encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache key."""
        # Use subdirectories to avoid too many files in one directory
        return self.cache_dir / cache_key[:2] / cache_key[2:4] / f"{cache_key}.pkl"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Parameters
        ----------
        key : str
            Cache key
            
        Returns
        -------
        Any or None
            Cached value or None if not found/expired
        """
        cache_key = self._get_cache_key(key)
        
        # Check memory cache first
        if self._memory_cache is not None:
            if cache_key in self._memory_cache:
                entry = self._memory_cache[cache_key]
                if self._is_valid_entry(entry):
                    logger.debug(f"Cache hit (memory): {key}")
                    return entry["data"]
                else:
                    del self._memory_cache[cache_key]
        
        # Check disk cache
        if self.backend == "disk" or self.backend == "hybrid":
            cache_path = self._get_cache_path(cache_key)
            
            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        entry = pickle.load(f)
                    
                    if self._is_valid_entry(entry):
                        logger.debug(f"Cache hit (disk): {key}")
                        
                        # Update memory cache if hybrid
                        if self._memory_cache is not None:
                            self._memory_cache[cache_key] = entry
                        
                        return entry["data"]
                    else:
                        # Remove expired entry
                        cache_path.unlink()
                        if cache_key in self.metadata:
                            del self.metadata[cache_key]
                            self._save_metadata()
                        
                except Exception as e:
                    logger.warning(f"Failed to load cache entry: {e}")
        
        logger.debug(f"Cache miss: {key}")
        return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache.
        
        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache
        """
        cache_key = self._get_cache_key(key)
        timestamp = datetime.now().isoformat()
        
        entry = {
            "data": value,
            "timestamp": timestamp,
            "key": key
        }
        
        # Memory cache
        if self._memory_cache is not None:
            self._memory_cache[cache_key] = entry
        
        # Disk cache
        if self.backend == "disk" or self.backend == "hybrid":
            cache_path = self._get_cache_path(cache_key)
            
            try:
                # Create parent directories
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save data
                with open(cache_path, "wb") as f:
                    pickle.dump(entry, f)
                
                # Update metadata
                self.metadata[cache_key] = {
                    "timestamp": timestamp,
                    "size": cache_path.stat().st_size,
                    "key": key
                }
                self._save_metadata()
                
                logger.debug(f"Cached: {key}")
                
                # Check size limit
                self._enforce_size_limit()
                
            except Exception as e:
                logger.warning(f"Failed to cache entry: {e}")
    
    def delete(self, key: str) -> None:
        """Delete entry from cache."""
        cache_key = self._get_cache_key(key)
        
        # Memory cache
        if self._memory_cache is not None and cache_key in self._memory_cache:
            del self._memory_cache[cache_key]
        
        # Disk cache
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            cache_path.unlink()
        
        # Metadata
        if cache_key in self.metadata:
            del self.metadata[cache_key]
            self._save_metadata()
    
    def clear(self) -> None:
        """Clear entire cache."""
        logger.info("Clearing cache")
        
        # Memory cache
        if self._memory_cache is not None:
            self._memory_cache.clear()
        
        # Disk cache
        for cache_file in self.cache_dir.rglob("*.pkl"):
            cache_file.unlink()
        
        # Metadata
        self.metadata.clear()
        self._save_metadata()
    
    def _is_valid_entry(self, entry: dict) -> bool:
        """Check if cache entry is still valid."""
        if "timestamp" not in entry:
            return False
        
        timestamp = datetime.fromisoformat(entry["timestamp"])
        age = datetime.now() - timestamp
        
        return age < self.ttl
    
    def _enforce_size_limit(self) -> None:
        """Remove old entries if cache size exceeds limit."""
        if self.backend not in ["disk", "hybrid"]:
            return
        
        # Calculate total size
        total_size = sum(
            entry.get("size", 0) for entry in self.metadata.values()
        )
        total_size_mb = total_size / (1024 * 1024)
        
        if total_size_mb > self.max_size_mb:
            logger.info(f"Cache size ({total_size_mb:.1f} MB) exceeds limit, cleaning up")
            
            # Sort by timestamp (oldest first)
            sorted_entries = sorted(
                self.metadata.items(),
                key=lambda x: x[1].get("timestamp", "")
            )
            
            # Remove oldest entries until under limit
            for cache_key, entry in sorted_entries:
                if total_size_mb <= self.max_size_mb * 0.9:  # 90% to avoid frequent cleanup
                    break
                
                cache_path = self._get_cache_path(cache_key)
                if cache_path.exists():
                    cache_path.unlink()
                
                total_size_mb -= entry.get("size", 0) / (1024 * 1024)
                del self.metadata[cache_key]
            
            self._save_metadata()
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_entries = len(self.metadata)
        total_size = sum(
            entry.get("size", 0) for entry in self.metadata.values()
        )
        
        # Count expired entries
        expired = 0
        for entry in self.metadata.values():
            if "timestamp" in entry:
                timestamp = datetime.fromisoformat(entry["timestamp"])
                age = datetime.now() - timestamp
                if age >= self.ttl:
                    expired += 1
        
        return {
            "total_entries": total_entries,
            "expired_entries": expired,
            "total_size_mb": total_size / (1024 * 1024),
            "max_size_mb": self.max_size_mb,
            "ttl_days": self.ttl.days,
            "backend": self.backend
        }