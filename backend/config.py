#!/usr/bin/env python3
"""
File: backend/config.py
Description: Configuration management for the Emotion-Driven Data Landscape application
Author: Dennis Smaltz
Acknowledgement: Claude Opus 4
Created: 2024
Python Version: 3.8+

This module provides:
- Environment-based configuration
- Configuration validation
- Secure handling of sensitive data
- Dynamic configuration updates
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import timedelta
import secrets
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    
    # Look for .env file in parent directory
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")
    else:
        logger.info("No .env file found, using system environment")
except ImportError:
    logger.info("python-dotenv not installed, using system environment only")

class ConfigError(Exception):
    """Configuration error"""
    pass

@dataclass
class APIConfig:
    """API-related configuration"""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    testing: bool = False
    
    # Security
    secret_key: str = field(default_factory=lambda: secrets.token_hex(32))
    
    # CORS settings
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:*", "https://localhost:*"])
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_default: str = "200 per day"
    rate_limit_storage_url: Optional[str] = None
    
    # Request settings
    max_content_length: int = 16 * 1024 * 1024  # 16MB
    request_timeout: int = 30  # seconds

@dataclass
class ModelConfig:
    """ML model configuration"""
    model_cache_dir: str = "./models"
    
    # Model names
    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"
    
    # Processing settings
    max_text_length: int = 512
    batch_size: int = 32
    
    # Device settings
    use_gpu: bool = True
    device: Optional[str] = None  # Auto-detect if None
    
    # Model loading
    load_in_8bit: bool = False
    model_revision: str = "main"

@dataclass
class CacheConfig:
    """Cache configuration"""
    # Memory cache
    memory_cache_enabled: bool = True
    memory_cache_size: int = 1000
    
    # File cache
    file_cache_enabled: bool = True
    file_cache_dir: str = "../data/cache/file_cache"
    
    # Redis cache
    redis_enabled: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    
    # TTL settings (seconds)
    default_ttl: int = 3600  # 1 hour
    api_response_ttl: int = 3600
    model_result_ttl: int = 86400  # 24 hours
    static_data_ttl: int = 604800  # 7 days

@dataclass
class DataSourceConfig:
    """Data source configuration"""
    # Rate limiting per source
    rate_limits: Dict[str, int] = field(default_factory=lambda: {
        "news": 60,      # requests per minute
        "reddit": 30,
        "quotes": 120,
        "wikipedia": 60
    })
    
    # API keys (optional)
    news_api_key: Optional[str] = None
    twitter_bearer_token: Optional[str] = None
    
    # Source-specific settings
    reddit_user_agent: str = "EmotionLandscape/1.0"
    max_items_per_source: int = 100
    
    # Fallback settings
    use_fallback_data: bool = True
    fallback_cache_ttl: int = 3600

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File logging
    file_logging_enabled: bool = True
    log_file_path: str = "../logs/app.log"
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # Special loggers
    suppress_loggers: List[str] = field(default_factory=lambda: [
        "urllib3.connectionpool",
        "transformers.tokenization_utils"
    ])

@dataclass
class SecurityConfig:
    """Security configuration"""
    # Authentication
    auth_enabled: bool = False
    jwt_secret_key: Optional[str] = field(default_factory=lambda: secrets.token_hex(32))
    jwt_algorithm: str = "HS256"
    jwt_expiration_delta: int = 86400  # 24 hours
    
    # API keys
    require_api_key: bool = False
    api_keys: List[str] = field(default_factory=list)
    
    # Content security
    allowed_text_types: List[str] = field(default_factory=lambda: [
        "text/plain", "text/html", "application/json"
    ])
    max_text_size: int = 1024 * 1024  # 1MB
    
    # Sanitization
    sanitize_input: bool = True
    strip_html: bool = True

class Config:
    """Main configuration class"""
    
    def __init__(self):
        # Load environment
        self.env = os.environ.get('FLASK_ENV', 'development')
        self.is_production = self.env == 'production'
        self.is_development = self.env == 'development'
        self.is_testing = self.env == 'testing'
        
        # Initialize sub-configurations
        self.api = self._load_api_config()
        self.models = self._load_model_config()
        self.cache = self._load_cache_config()
        self.data_sources = self._load_data_source_config()
        self.logging = self._load_logging_config()
        self.security = self._load_security_config()
        
        # Flask-specific settings
        self.SECRET_KEY = self.api.secret_key
        self.DEBUG = self.api.debug
        self.TESTING = self.api.testing
        self.MAX_CONTENT_LENGTH = self.api.max_content_length
        
        # CORS settings
        self.ALLOWED_ORIGINS = self.api.cors_origins
        
        # Additional settings
        self.PROPAGATE_EXCEPTIONS = True
        self.JSON_SORT_KEYS = False
        self.JSONIFY_PRETTYPRINT_REGULAR = self.is_development
        
        # Validate configuration
        self._validate_config()
        
        # Setup logging
        self._setup_logging()
    
    def _load_api_config(self) -> APIConfig:
        """Load API configuration from environment"""
        return APIConfig(
            host=os.environ.get('API_HOST', '0.0.0.0'),
            port=int(os.environ.get('API_PORT', 5000)),
            debug=self.is_development,
            testing=self.is_testing,
            secret_key=os.environ.get('SECRET_KEY', secrets.token_hex(32)),
            cors_origins=self._parse_list(os.environ.get('CORS_ORIGINS', 'http://localhost:*'))
        )
    
    def _load_model_config(self) -> ModelConfig:
        """Load model configuration from environment"""
        return ModelConfig(
            model_cache_dir=os.environ.get('MODEL_CACHE_DIR', './models'),
            sentiment_model=os.environ.get('SENTIMENT_MODEL', ModelConfig.sentiment_model),
            emotion_model=os.environ.get('EMOTION_MODEL', ModelConfig.emotion_model),
            max_text_length=int(os.environ.get('MAX_TEXT_LENGTH', 512)),
            batch_size=int(os.environ.get('BATCH_SIZE', 32)),
            use_gpu=os.environ.get('USE_GPU', 'true').lower() == 'true'
        )
    
    def _load_cache_config(self) -> CacheConfig:
        """Load cache configuration from environment"""
        return CacheConfig(
            memory_cache_enabled=os.environ.get('MEMORY_CACHE_ENABLED', 'true').lower() == 'true',
            memory_cache_size=int(os.environ.get('MEMORY_CACHE_SIZE', 1000)),
            file_cache_enabled=os.environ.get('FILE_CACHE_ENABLED', 'true').lower() == 'true',
            file_cache_dir=os.environ.get('FILE_CACHE_DIR', '../data/cache/file_cache'),
            redis_enabled=os.environ.get('ENABLE_REDIS', 'false').lower() == 'true',
            redis_host=os.environ.get('REDIS_HOST', 'localhost'),
            redis_port=int(os.environ.get('REDIS_PORT', 6379)),
            redis_password=os.environ.get('REDIS_PASSWORD'),
            default_ttl=int(os.environ.get('CACHE_TTL', 3600))
        )
    
    def _load_data_source_config(self) -> DataSourceConfig:
        """Load data source configuration from environment"""
        return DataSourceConfig(
            news_api_key=os.environ.get('NEWS_API_KEY'),
            twitter_bearer_token=os.environ.get('TWITTER_BEARER_TOKEN'),
            reddit_user_agent=os.environ.get('REDDIT_USER_AGENT', 'EmotionLandscape/1.0 (https://github.com/dnoice/emotion-landscape)'),
            use_fallback_data=os.environ.get('USE_FALLBACK_DATA', 'true').lower() == 'true'
        )
    
    def _load_logging_config(self) -> LoggingConfig:
        """Load logging configuration from environment"""
        return LoggingConfig(
            level=os.environ.get('LOG_LEVEL', 'INFO'),
            file_logging_enabled=os.environ.get('FILE_LOGGING_ENABLED', 'true').lower() == 'true',
            log_file_path=os.environ.get('LOG_FILE_PATH', '../logs/app.log')
        )
    
    def _load_security_config(self) -> SecurityConfig:
        """Load security configuration from environment"""
        api_keys = self._parse_list(os.environ.get('API_KEYS', ''))
        
        return SecurityConfig(
            auth_enabled=os.environ.get('AUTH_ENABLED', 'false').lower() == 'true',
            jwt_secret_key=os.environ.get('JWT_SECRET_KEY', secrets.token_hex(32)),
            require_api_key=bool(api_keys),
            api_keys=api_keys,
            sanitize_input=os.environ.get('SANITIZE_INPUT', 'true').lower() == 'true'
        )
    
    def _parse_list(self, value: str) -> List[str]:
        """Parse comma-separated list from environment variable"""
        if not value:
            return []
        return [item.strip() for item in value.split(',') if item.strip()]
    
    def _validate_config(self):
        """Validate configuration values"""
        errors = []
        
        # Validate port
        if not 1 <= self.api.port <= 65535:
            errors.append(f"Invalid port: {self.api.port}")
        
        # Validate cache settings
        if self.cache.redis_enabled and not self.cache.redis_host:
            errors.append("Redis enabled but no host specified")
        
        # Validate model settings
        if self.models.batch_size < 1:
            errors.append(f"Invalid batch size: {self.models.batch_size}")
        
        # Check for required API keys in production
        if self.is_production:
            if not self.api.secret_key or self.api.secret_key == 'dev-secret-key':
                errors.append("Insecure SECRET_KEY in production")
            
            if self.api.debug:
                errors.append("Debug mode enabled in production")
        
        if errors:
            raise ConfigError(f"Configuration errors: {'; '.join(errors)}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Set logging level
        log_level = getattr(logging, self.logging.level.upper(), logging.INFO)
        logging.getLogger().setLevel(log_level)
        
        # Configure file logging
        if self.logging.file_logging_enabled:
            from logging.handlers import RotatingFileHandler
            
            # Create log directory
            log_dir = Path(self.logging.log_file_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create rotating file handler
            file_handler = RotatingFileHandler(
                self.logging.log_file_path,
                maxBytes=self.logging.max_log_size,
                backupCount=self.logging.backup_count
            )
            
            file_handler.setFormatter(
                logging.Formatter(self.logging.format)
            )
            
            logging.getLogger().addHandler(file_handler)
        
        # Suppress noisy loggers
        for logger_name in self.logging.suppress_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration as dictionary"""
        return {
            'host': self.cache.redis_host,
            'port': self.cache.redis_port,
            'db': self.cache.redis_db,
            'password': self.cache.redis_password,
            'ssl': self.cache.redis_ssl
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding sensitive data)"""
        config_dict = {
            'env': self.env,
            'api': asdict(self.api),
            'models': asdict(self.models),
            'cache': asdict(self.cache),
            'data_sources': asdict(self.data_sources),
            'logging': asdict(self.logging)
        }
        
        # Remove sensitive data
        config_dict['api']['secret_key'] = '***'
        config_dict['cache']['redis_password'] = '***' if self.cache.redis_password else None
        config_dict['data_sources']['news_api_key'] = '***' if self.data_sources.news_api_key else None
        config_dict['data_sources']['twitter_bearer_token'] = '***' if self.data_sources.twitter_bearer_token else None
        
        return config_dict
    
    def save_to_file(self, filepath: str):
        """Save configuration to file (for debugging)"""
        config_dict = self.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    @lru_cache(maxsize=1)
    def get_instance(cls) -> 'Config':
        """Get singleton instance of configuration"""
        return cls()

# Create example .env file
def create_env_example():
    """Create an example .env file"""
    env_example = """# Emotion-Driven Data Landscape Configuration
# Copy this file to .env and update with your values

# Environment (development, production, testing)
FLASK_ENV=development

# API Settings
API_HOST=0.0.0.0
API_PORT=5000
SECRET_KEY=your-secret-key-here

# CORS Origins (comma-separated)
CORS_ORIGINS=http://localhost:8000,http://localhost:3000

# Model Settings
MODEL_CACHE_DIR=./models
MAX_TEXT_LENGTH=512
BATCH_SIZE=32
USE_GPU=true

# Cache Settings
MEMORY_CACHE_ENABLED=true
MEMORY_CACHE_SIZE=1000
FILE_CACHE_ENABLED=true
FILE_CACHE_DIR=../data/cache/file_cache
CACHE_TTL=3600

# Redis Settings (optional)
ENABLE_REDIS=false
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Data Source Settings
REDDIT_USER_AGENT=EmotionLandscape/1.0
USE_FALLBACK_DATA=true

# API Keys (optional)
NEWS_API_KEY=
TWITTER_BEARER_TOKEN=

# Logging
LOG_LEVEL=INFO
FILE_LOGGING_ENABLED=true
LOG_FILE_PATH=../logs/app.log

# Security
AUTH_ENABLED=false
JWT_SECRET_KEY=
SANITIZE_INPUT=true

# API Keys for rate limiting (comma-separated)
API_KEYS=
"""
    
    with open('.env.example', 'w') as f:
        f.write(env_example)
    
    print("Created .env.example file")

# Convenience function to get config
def get_config() -> Config:
    """Get the configuration instance"""
    return Config.get_instance()

# Example usage
if __name__ == "__main__":
    # Create example env file
    create_env_example()
    
    # Load and display configuration
    config = get_config()
    
    print("Configuration loaded successfully!")
    print(f"Environment: {config.env}")
    print(f"API Port: {config.api.port}")
    print(f"Debug Mode: {config.api.debug}")
    print(f"Cache TTL: {config.cache.default_ttl}s")
    print(f"Redis Enabled: {config.cache.redis_enabled}")
    
    # Save configuration for debugging
    config.save_to_file("config_debug.json")
    
    # Display full configuration (excluding sensitive data)
    print("\nFull configuration:")
    print(json.dumps(config.to_dict(), indent=2))
