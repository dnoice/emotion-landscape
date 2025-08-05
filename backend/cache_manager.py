#!/usr/bin/env python3
"""
File: backend/cache_manager.py
Description: Intelligent caching system for API responses and processed data
Author: Dennis Smaltz
Acknowledgement: Claude Opus 4
Created: 2024
Python Version: 3.8+

This module provides:
- In-memory caching with TTL
- File-based persistent caching
- Redis support (optional)
- Cache warming and invalidation
- Memory management
"""

import os
import json
import time
import pickle
import hashlib
import threading
import logging
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple, Callable
from datetime import datetime, timedelta
from collections import OrderedDict
from functools import wraps
import atexit
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Redis (optional dependency)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.info("Redis not available. Using file-based caching only.")

class CacheStats:
    """Track cache performance statistics"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.errors = 0
        self.total_size = 0
        self.lock = threading.Lock()
    
    def record_hit(self):
        with self.lock:
            self.hits += 1
    
    def record_miss(self):
        with self.lock:
            self.misses += 1
    
    def record_eviction(self):
        with self.lock:
            self.evictions += 1
    
    def record_error(self):
        with self.lock:
            self.errors += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0
            
            return {
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'errors': self.errors,
                'total_requests': total,
                'hit_rate': f"{hit_rate:.2f}%",
                'total_size': self.total_size
            }

class LRUCache:
    """Thread-safe LRU cache implementation"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.stats = CacheStats()
    
    def get(self, key: str) -> Optional[Tuple[Any, float]]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.stats.record_hit()
                return self.cache[key]
            
            self.stats.record_miss()
            return None
    
    def set(self, key: str, value: Any, expiry: float):
        """Set value in cache"""
        with self.lock:
            # Remove oldest items if at capacity
            while len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.stats.record_eviction()
            
            self.cache[key] = (value, expiry)
            self.cache.move_to_end(key)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
    
    def cleanup_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        with self.lock:
            expired_keys = [
                key for key, (_, expiry) in self.cache.items()
                if expiry < current_time
            ]
            
            for key in expired_keys:
                del self.cache[key]
                self.stats.record_eviction()
    
    def get_size(self) -> int:
        """Get number of cached items"""
        return len(self.cache)

class CacheManager:
    """
    Comprehensive cache management system
    Supports memory, file, and Redis caching
    """
    
    def __init__(self, 
                 cache_dir: str = "../data/cache",
                 memory_cache_size: int = 1000,
                 default_ttl: int = 3600,
                 enable_redis: bool = False,
                 redis_config: Dict[str, Any] = None):
        
        # Configuration
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        
        # Memory cache
        self.memory_cache = LRUCache(max_size=memory_cache_size)
        
        # File cache settings
        self.file_cache_enabled = True
        self.file_cache_dir = self.cache_dir / "file_cache"
        self.file_cache_dir.mkdir(exist_ok=True)
        
        # Redis cache
        self.redis_client = None
        self.redis_enabled = enable_redis and REDIS_AVAILABLE
        
        if self.redis_enabled:
            self._init_redis(redis_config or {})
        
        # Background cleanup thread
        self.cleanup_interval = 300  # 5 minutes
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        # Register cleanup on exit
        atexit.register(self.shutdown)
        
        logger.info(f"Cache manager initialized (memory_size={memory_cache_size}, "
                   f"file_cache={self.file_cache_enabled}, redis={self.redis_enabled})")
    
    def _init_redis(self, config: Dict[str, Any]):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=config.get('host', 'localhost'),
                port=config.get('port', 6379),
                db=config.get('db', 0),
                password=config.get('password'),
                decode_responses=False,  # We'll handle encoding
                socket_keepalive=True,
                socket_keepalive_options={}
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_enabled = False
            self.redis_client = None
    
    def _generate_key(self, key: str) -> str:
        """Generate a normalized cache key"""
        # Add prefix to avoid collisions
        prefixed_key = f"emotion_landscape:{key}"
        
        # Hash if too long
        if len(prefixed_key) > 250:
            hash_suffix = hashlib.md5(prefixed_key.encode()).hexdigest()[:8]
            prefixed_key = f"{prefixed_key[:240]}_{hash_suffix}"
        
        return prefixed_key
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache (checks all cache layers)
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        cache_key = self._generate_key(key)
        
        # 1. Check memory cache
        result = self.memory_cache.get(cache_key)
        if result:
            value, expiry = result
            if expiry > time.time():
                logger.debug(f"Memory cache hit: {key}")
                return value
            else:
                self.memory_cache.delete(cache_key)
        
        # 2. Check Redis cache
        if self.redis_enabled:
            try:
                redis_value = self.redis_client.get(cache_key)
                if redis_value:
                    value = pickle.loads(redis_value)
                    # Add to memory cache
                    ttl = self.redis_client.ttl(cache_key)
                    if ttl > 0:
                        expiry = time.time() + ttl
                        self.memory_cache.set(cache_key, value, expiry)
                    logger.debug(f"Redis cache hit: {key}")
                    return value
            except Exception as e:
                logger.error(f"Redis get error: {e}")
                self.memory_cache.stats.record_error()
        
        # 3. Check file cache
        if self.file_cache_enabled:
            file_value = self._get_from_file(cache_key)
            if file_value is not None:
                # Add to faster caches
                expiry = time.time() + self.default_ttl
                self.memory_cache.set(cache_key, file_value, expiry)
                if self.redis_enabled:
                    self._set_in_redis(cache_key, file_value, self.default_ttl)
                logger.debug(f"File cache hit: {key}")
                return file_value
        
        logger.debug(f"Cache miss: {key}")
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set value in cache (all layers)
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        cache_key = self._generate_key(key)
        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl
        
        # 1. Set in memory cache
        self.memory_cache.set(cache_key, value, expiry)
        
        # 2. Set in Redis cache
        if self.redis_enabled:
            self._set_in_redis(cache_key, value, ttl)
        
        # 3. Set in file cache
        if self.file_cache_enabled:
            self._set_in_file(cache_key, value, ttl)
        
        logger.debug(f"Cached: {key} (ttl={ttl}s)")
    
    def delete(self, key: str) -> bool:
        """Delete value from all cache layers"""
        cache_key = self._generate_key(key)
        deleted = False
        
        # Delete from memory
        if self.memory_cache.delete(cache_key):
            deleted = True
        
        # Delete from Redis
        if self.redis_enabled:
            try:
                if self.redis_client.delete(cache_key):
                    deleted = True
            except Exception as e:
                logger.error(f"Redis delete error: {e}")
        
        # Delete from file
        if self.file_cache_enabled:
            if self._delete_from_file(cache_key):
                deleted = True
        
        return deleted
    
    def clear(self, pattern: Optional[str] = None):
        """
        Clear cache entries
        
        Args:
            pattern: Optional pattern to match keys (supports * wildcard)
        """
        if pattern:
            # Clear matching pattern
            if '*' in pattern:
                # Convert to regex pattern
                import re
                regex_pattern = pattern.replace('*', '.*')
                regex = re.compile(regex_pattern)
                
                # Clear from memory cache
                with self.memory_cache.lock:
                    keys_to_delete = [
                        k for k in self.memory_cache.cache.keys()
                        if regex.match(k)
                    ]
                    for k in keys_to_delete:
                        del self.memory_cache.cache[k]
                
                # Clear from Redis
                if self.redis_enabled:
                    try:
                        for key in self.redis_client.scan_iter(match=f"*{pattern}*"):
                            self.redis_client.delete(key)
                    except Exception as e:
                        logger.error(f"Redis clear error: {e}")
            else:
                # Clear specific key
                self.delete(pattern)
        else:
            # Clear all
            self.memory_cache.clear()
            
            if self.redis_enabled:
                try:
                    self.redis_client.flushdb()
                except Exception as e:
                    logger.error(f"Redis flush error: {e}")
            
            if self.file_cache_enabled:
                # Clear file cache
                for file in self.file_cache_dir.glob("*.cache"):
                    try:
                        file.unlink()
                    except:
                        pass
        
        logger.info(f"Cache cleared (pattern={pattern})")
    
    def _set_in_redis(self, key: str, value: Any, ttl: int):
        """Set value in Redis"""
        try:
            serialized = pickle.dumps(value)
            self.redis_client.setex(key, ttl, serialized)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            self.memory_cache.stats.record_error()
    
    def _get_from_file(self, key: str) -> Optional[Any]:
        """Get value from file cache"""
        try:
            # Generate safe filename
            safe_key = hashlib.md5(key.encode()).hexdigest()
            file_path = self.file_cache_dir / f"{safe_key}.cache"
            
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Check expiry
                if data['expiry'] > time.time():
                    return data['value']
                else:
                    # Delete expired file
                    file_path.unlink()
            
        except Exception as e:
            logger.error(f"File cache read error: {e}")
            self.memory_cache.stats.record_error()
        
        return None
    
    def _set_in_file(self, key: str, value: Any, ttl: int):
        """Set value in file cache"""
        try:
            # Generate safe filename
            safe_key = hashlib.md5(key.encode()).hexdigest()
            file_path = self.file_cache_dir / f"{safe_key}.cache"
            
            data = {
                'key': key,
                'value': value,
                'expiry': time.time() + ttl,
                'created': datetime.utcnow().isoformat()
            }
            
            # Write atomically
            temp_path = file_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f)
            
            temp_path.replace(file_path)
            
        except Exception as e:
            logger.error(f"File cache write error: {e}")
            self.memory_cache.stats.record_error()
    
    def _delete_from_file(self, key: str) -> bool:
        """Delete from file cache"""
        try:
            safe_key = hashlib.md5(key.encode()).hexdigest()
            file_path = self.file_cache_dir / f"{safe_key}.cache"
            
            if file_path.exists():
                file_path.unlink()
                return True
                
        except Exception as e:
            logger.error(f"File cache delete error: {e}")
        
        return False
    
    def _cleanup_loop(self):
        """Background cleanup thread"""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                
                # Cleanup memory cache
                self.memory_cache.cleanup_expired()
                
                # Cleanup file cache
                if self.file_cache_enabled:
                    self._cleanup_file_cache()
                
                # Force garbage collection if memory usage is high
                import psutil
                process = psutil.Process()
                memory_percent = process.memory_percent()
                
                if memory_percent > 80:
                    logger.warning(f"High memory usage: {memory_percent:.1f}%")
                    gc.collect()
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    def _cleanup_file_cache(self):
        """Clean up expired file cache entries"""
        try:
            current_time = time.time()
            cleaned = 0
            
            for file_path in self.file_cache_dir.glob("*.cache"):
                try:
                    # Check file age
                    if file_path.stat().st_mtime < current_time - (7 * 24 * 3600):  # 7 days
                        file_path.unlink()
                        cleaned += 1
                        continue
                    
                    # Check content expiry
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    if data.get('expiry', 0) < current_time:
                        file_path.unlink()
                        cleaned += 1
                        
                except Exception:
                    # Delete corrupted files
                    try:
                        file_path.unlink()
                        cleaned += 1
                    except:
                        pass
            
            if cleaned > 0:
                logger.info(f"Cleaned {cleaned} expired file cache entries")
                
        except Exception as e:
            logger.error(f"File cache cleanup error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            'memory_cache': self.memory_cache.stats.get_stats(),
            'memory_cache_size': self.memory_cache.get_size(),
            'file_cache_enabled': self.file_cache_enabled,
            'redis_enabled': self.redis_enabled,
        }
        
        if self.file_cache_enabled:
            try:
                file_count = len(list(self.file_cache_dir.glob("*.cache")))
                stats['file_cache_count'] = file_count
            except:
                pass
        
        if self.redis_enabled:
            try:
                stats['redis_connected'] = self.redis_client.ping()
                stats['redis_db_size'] = self.redis_client.dbsize()
            except:
                stats['redis_connected'] = False
        
        return stats
    
    def warm_cache(self, items: List[Tuple[str, Any, Optional[int]]]):
        """
        Warm cache with pre-computed values
        
        Args:
            items: List of (key, value, ttl) tuples
        """
        logger.info(f"Warming cache with {len(items)} items")
        
        for key, value, ttl in items:
            self.set(key, value, ttl)
    
    def shutdown(self):
        """Cleanup resources"""
        logger.info("Shutting down cache manager")
        
        # Save memory cache stats
        stats = self.get_stats()
        logger.info(f"Final cache stats: {stats}")
        
        # Close Redis connection
        if self.redis_client:
            try:
                self.redis_client.close()
            except:
                pass

# Decorator for caching function results
def cached(ttl: Optional[int] = None, 
          key_prefix: Optional[str] = None,
          cache_manager: Optional[CacheManager] = None):
    """
    Decorator to cache function results
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Optional prefix for cache keys
        cache_manager: CacheManager instance (uses global if not provided)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix or func.__name__]
            
            # Add args to key
            for arg in args:
                if isinstance(arg, (str, int, float, bool)):
                    key_parts.append(str(arg))
                else:
                    key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:8])
            
            # Add kwargs to key
            for k, v in sorted(kwargs.items()):
                if isinstance(v, (str, int, float, bool)):
                    key_parts.append(f"{k}={v}")
                else:
                    key_parts.append(f"{k}={hashlib.md5(str(v).encode()).hexdigest()[:8]}")
            
            cache_key = ":".join(key_parts)
            
            # Get cache manager
            cm = cache_manager or getattr(wrapper, '_cache_manager', None)
            if not cm:
                # Create default cache manager
                cm = CacheManager()
                wrapper._cache_manager = cm
            
            # Check cache
            cached_result = cm.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result
            cm.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator

# Example usage
if __name__ == "__main__":
    # Test cache manager
    cache = CacheManager(
        memory_cache_size=100,
        default_ttl=60,
        enable_redis=False  # Set to True if Redis is available
    )
    
    # Test basic operations
    print("Testing cache operations...")
    
    # Set and get
    cache.set("test_key", {"data": "test_value"}, ttl=30)
    result = cache.get("test_key")
    print(f"Get result: {result}")
    
    # Test decorator
    @cached(ttl=60, cache_manager=cache)
    def expensive_function(x, y):
        print(f"Computing {x} + {y}...")
        time.sleep(1)  # Simulate expensive operation
        return x + y
    
    # First call (cache miss)
    result1 = expensive_function(5, 3)
    print(f"Result 1: {result1}")
    
    # Second call (cache hit)
    result2 = expensive_function(5, 3)
    print(f"Result 2: {result2}")
    
    # Get stats
    stats = cache.get_stats()
    print(f"\nCache stats: {json.dumps(stats, indent=2)}")
    
    # Cleanup
    cache.shutdown()
