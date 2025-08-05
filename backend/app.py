#!/usr/bin/env python3
"""
File: backend/app.py
Description: Main Flask application server for Emotion-Driven Data Landscape
Author: Dennis Smaltz
Acknowledgment: Claude Opus 4
Created: 2024
Python Version: 3.8+

This is the main entry point for the backend API. It provides endpoints for:
- Analyzing text data from various sources
- Retrieving available data sources
- Real-time emotion analysis
- Trend analysis over time
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, List, Any, Optional

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import numpy as np

# Import our modules
from ml_models import SentimentAnalyzer, EmotionDetector, TopicModeler, setup_models
from data_processor import DataProcessor
from cache_manager import CacheManager
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Configure CORS
CORS(app, resources={
    r"/api/*": {
        "origins": Config.ALLOWED_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Initialize components
logger.info("Initializing ML models...")
sentiment_analyzer = SentimentAnalyzer()
emotion_detector = EmotionDetector()
topic_modeler = TopicModeler()

logger.info("Initializing data processor and cache...")
data_processor = DataProcessor()
cache_manager = CacheManager()

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found',
        'code': 404
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'status': 'error',
        'message': 'Internal server error',
        'code': 500
    }), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        'status': 'error',
        'message': 'Rate limit exceeded',
        'code': 429,
        'retry_after': e.description
    }), 429

# Utility decorators
def validate_json(f):
    """Decorator to validate JSON input"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method == 'POST':
            if not request.is_json:
                return jsonify({
                    'status': 'error',
                    'message': 'Content-Type must be application/json'
                }), 400
        return f(*args, **kwargs)
    return decorated_function

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'models': {
            'sentiment': sentiment_analyzer.is_ready(),
            'emotion': emotion_detector.is_ready(),
            'topic': topic_modeler.is_ready()
        }
    })

@app.route('/api/analyze', methods=['GET'])
@limiter.limit("30 per minute")
def analyze_data():
    """
    Fetch and analyze data from various sources
    
    Query Parameters:
    - source: Data source (news, reddit, quotes, etc.)
    - limit: Number of items to analyze (default: 50, max: 100)
    - cache: Whether to use cached results (default: true)
    """
    try:
        # Get parameters
        source = request.args.get('source', 'news')
        limit = min(int(request.args.get('limit', 50)), 100)
        use_cache = request.args.get('cache', 'true').lower() == 'true'
        
        # Check cache
        cache_key = f"analyze:{source}:{limit}"
        if use_cache:
            cached_data = cache_manager.get(cache_key)
            if cached_data:
                logger.info(f"Returning cached data for {cache_key}")
                return jsonify(cached_data)
        
        # Fetch data
        logger.info(f"Fetching {limit} items from {source}")
        raw_data = data_processor.fetch_data(source, limit=limit)
        
        if not raw_data:
            return jsonify({
                'status': 'error',
                'message': f'No data available from source: {source}'
            }), 404
        
        # Analyze data
        results = []
        texts = [item['text'] for item in raw_data]
        
        # Batch process for efficiency
        logger.info(f"Analyzing {len(texts)} texts...")
        sentiments = sentiment_analyzer.analyze_batch(texts)
        emotions = emotion_detector.detect_batch(texts)
        
        # Extract topics if enough data
        topics = []
        if len(texts) >= 10:
            topics = topic_modeler.extract_topics(texts, num_topics=5)
        
        # Combine results
        for i, item in enumerate(raw_data):
            results.append({
                'id': item.get('id', f"{source}_{i}"),
                'text': item['text'][:200] + '...' if len(item['text']) > 200 else item['text'],
                'source': item.get('source', source),
                'url': item.get('url', ''),
                'sentiment': sentiments[i],
                'emotions': emotions[i],
                'timestamp': item.get('timestamp', datetime.utcnow().isoformat()),
                'metadata': {
                    'intensity': calculate_intensity(sentiments[i], emotions[i]),
                    'x': map_to_coordinate(sentiments[i]['score'], -1, 1),
                    'y': map_to_coordinate(get_emotion_valence(emotions[i]), -1, 1)
                }
            })
        
        # Calculate summary statistics
        summary = calculate_summary(results, topics)
        
        response_data = {
            'status': 'success',
            'data': results,
            'summary': summary,
            'topics': topics,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Cache the results
        if use_cache:
            cache_manager.set(cache_key, response_data, ttl=3600)  # 1 hour
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in analyze_data: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'Failed to analyze data',
            'details': str(e) if app.debug else None
        }), 500

@app.route('/api/analyze/text', methods=['POST'])
@validate_json
@limiter.limit("10 per minute")
def analyze_custom_text():
    """
    Analyze custom text provided by the user
    
    Request Body:
    {
        "text": "Text to analyze",
        "include_topics": true/false (optional)
    }
    """
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'status': 'error',
                'message': 'Text field is required'
            }), 400
        
        if len(text) > 5000:
            return jsonify({
                'status': 'error',
                'message': 'Text too long (max 5000 characters)'
            }), 400
        
        # Analyze the text
        sentiment = sentiment_analyzer.analyze(text)
        emotions = emotion_detector.detect(text)
        
        result = {
            'status': 'success',
            'analysis': {
                'sentiment': sentiment,
                'emotions': emotions,
                'metadata': {
                    'intensity': calculate_intensity(sentiment, emotions),
                    'text_length': len(text),
                    'word_count': len(text.split())
                }
            }
        }
        
        # Optionally include topic analysis
        if data.get('include_topics', False) and len(text.split()) > 20:
            topics = topic_modeler.extract_topics([text], num_topics=3)
            result['analysis']['topics'] = topics
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in analyze_custom_text: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'Failed to analyze text'
        }), 500

@app.route('/api/sources', methods=['GET'])
def get_sources():
    """Get available data sources with metadata"""
    sources = [
        {
            'id': 'news',
            'name': 'News Headlines',
            'description': 'Latest news from major outlets',
            'icon': '📰',
            'update_frequency': 'hourly',
            'available': True
        },
        {
            'id': 'reddit',
            'name': 'Reddit Posts',
            'description': 'Trending posts from Reddit',
            'icon': '💬',
            'update_frequency': 'real-time',
            'available': True
        },
        {
            'id': 'quotes',
            'name': 'Famous Quotes',
            'description': 'Inspirational and thought-provoking quotes',
            'icon': '💭',
            'update_frequency': 'daily',
            'available': True
        },
        {
            'id': 'wikipedia',
            'name': 'Wikipedia Current Events',
            'description': 'Current events from Wikipedia',
            'icon': '📚',
            'update_frequency': 'daily',
            'available': True
        }
    ]
    
    return jsonify({
        'status': 'success',
        'sources': sources
    })

@app.route('/api/trends', methods=['GET'])
@limiter.limit("10 per minute")
def get_trends():
    """
    Get emotion and sentiment trends over time
    
    Query Parameters:
    - period: Time period (hour, day, week)
    - source: Data source filter (optional)
    """
    try:
        period = request.args.get('period', 'day')
        source = request.args.get('source', None)
        
        # For demo purposes, generate sample trend data
        # In production, this would query a time-series database
        trends = generate_sample_trends(period, source)
        
        return jsonify({
            'status': 'success',
            'trends': trends,
            'period': period,
            'source': source
        })
        
    except Exception as e:
        logger.error(f"Error in get_trends: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'Failed to get trends'
        }), 500

@app.route('/api/stream', methods=['GET'])
def stream_updates():
    """
    Server-sent events endpoint for real-time updates
    """
    def generate():
        """Generate server-sent events"""
        while True:
            # Get latest data (simplified for demo)
            data = {
                'timestamp': datetime.utcnow().isoformat(),
                'emotion_pulse': {
                    'joy': np.random.random(),
                    'sadness': np.random.random(),
                    'anger': np.random.random(),
                    'fear': np.random.random(),
                    'surprise': np.random.random()
                }
            }
            
            yield f"data: {json.dumps(data)}\n\n"
            
            # Wait before next update
            import time
            time.sleep(5)
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )

# Helper functions
def calculate_intensity(sentiment: Dict, emotions: Dict) -> float:
    """Calculate overall emotional intensity"""
    sentiment_intensity = abs(sentiment['score'])
    emotion_intensity = max(emotions.values()) if emotions else 0
    return (sentiment_intensity + emotion_intensity) / 2

def get_emotion_valence(emotions: Dict) -> float:
    """Calculate emotional valence (positive vs negative)"""
    positive_emotions = ['joy', 'surprise']
    negative_emotions = ['sadness', 'anger', 'fear', 'disgust']
    
    positive_score = sum(emotions.get(e, 0) for e in positive_emotions)
    negative_score = sum(emotions.get(e, 0) for e in negative_emotions)
    
    total = positive_score + negative_score
    if total == 0:
        return 0
    
    return (positive_score - negative_score) / total

def map_to_coordinate(value: float, min_val: float, max_val: float) -> float:
    """Map a value to coordinate system"""
    return (value - min_val) / (max_val - min_val) * 2 - 1 if max_val != min_val else 0

def calculate_summary(results: List[Dict], topics: List[Dict]) -> Dict:
    """Calculate aggregate statistics from results"""
    if not results:
        return {}
    
    sentiments = [r['sentiment']['score'] for r in results]
    emotions = {}
    
    for r in results:
        for emotion, score in r['emotions'].items():
            if emotion not in emotions:
                emotions[emotion] = []
            emotions[emotion].append(score)
    
    # Calculate emotion stats
    emotion_stats = {}
    dominant_emotion = None
    max_emotion_score = 0
    
    for emotion, scores in emotions.items():
        avg_score = np.mean(scores)
        emotion_stats[emotion] = {
            'average': avg_score,
            'std': np.std(scores),
            'min': min(scores),
            'max': max(scores)
        }
        
        if avg_score > max_emotion_score:
            max_emotion_score = avg_score
            dominant_emotion = emotion
    
    return {
        'total_items': len(results),
        'sentiment': {
            'average': np.mean(sentiments),
            'std': np.std(sentiments),
            'positive_ratio': sum(1 for s in sentiments if s > 0.1) / len(sentiments),
            'negative_ratio': sum(1 for s in sentiments if s < -0.1) / len(sentiments),
            'neutral_ratio': sum(1 for s in sentiments if -0.1 <= s <= 0.1) / len(sentiments)
        },
        'emotions': emotion_stats,
        'dominant_emotion': dominant_emotion,
        'emotional_diversity': calculate_diversity(emotions),
        'top_keywords': extract_top_keywords(results) if not topics else None
    }

def calculate_diversity(emotions: Dict[str, List[float]]) -> float:
    """Calculate emotional diversity using entropy"""
    avg_emotions = [np.mean(scores) for scores in emotions.values()]
    total = sum(avg_emotions)
    
    if total == 0:
        return 0
    
    probabilities = [e / total for e in avg_emotions]
    entropy = -sum(p * np.log(p) for p in probabilities if p > 0)
    
    # Normalize to 0-1 range
    max_entropy = np.log(len(emotions))
    return entropy / max_entropy if max_entropy > 0 else 0

def extract_top_keywords(results: List[Dict], top_n: int = 10) -> List[str]:
    """Extract top keywords from results (simplified)"""
    # In production, use TF-IDF or similar
    word_counts = {}
    
    for r in results:
        words = r['text'].lower().split()
        for word in words:
            if len(word) > 4:  # Simple filter
                word_counts[word] = word_counts.get(word, 0) + 1
    
    return sorted(word_counts.keys(), key=word_counts.get, reverse=True)[:top_n]

def generate_sample_trends(period: str, source: Optional[str]) -> List[Dict]:
    """Generate sample trend data for demo"""
    now = datetime.utcnow()
    points = 24 if period == 'hour' else 7 if period == 'day' else 4
    
    trends = []
    for i in range(points):
        if period == 'hour':
            timestamp = now - timedelta(hours=i)
        elif period == 'day':
            timestamp = now - timedelta(days=i)
        else:
            timestamp = now - timedelta(weeks=i)
        
        trends.append({
            'timestamp': timestamp.isoformat(),
            'sentiment': {
                'average': np.random.uniform(-0.5, 0.5),
                'volume': np.random.randint(100, 1000)
            },
            'emotions': {
                'joy': np.random.random(),
                'sadness': np.random.random(),
                'anger': np.random.random(),
                'fear': np.random.random(),
                'surprise': np.random.random()
            }
        })
    
    return list(reversed(trends))

# Application startup
def initialize_app():
    """Initialize application components"""
    logger.info("Setting up ML models...")
    setup_models()
    
    logger.info("Warming up cache...")
    # Pre-fetch some data for common sources
    for source in ['news', 'quotes']:
        try:
            data_processor.fetch_data(source, limit=10)
        except Exception as e:
            logger.warning(f"Failed to pre-fetch {source}: {e}")
    
    logger.info("Application initialized successfully")

# Main entry point
if __name__ == '__main__':
    initialize_app()
    
    # Run the application
    port = int(os.environ.get('API_PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Flask app on port {port} (debug={debug})")
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )
