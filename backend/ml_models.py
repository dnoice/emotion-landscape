#!/usr/bin/env python3
"""
File: backend/ml_models.py
Description: Machine Learning models for sentiment analysis, emotion detection, and topic modeling
Author: Dennis Smaltz
Acknowledgement: Claude Opus 4
Created: 2024
Python Version: 3.8+

This module provides:
- Sentiment Analysis using DistilBERT
- Emotion Detection using RoBERTa
- Topic Modeling using LDA
- Model management and caching
"""

import os
import sys
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import warnings

import numpy as np
import torch
from transformers import (
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
@dataclass
class ModelConfig:
    """Configuration for ML models"""
    model_cache_dir: str = "./models"
    max_length: int = 512
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    
    # Model names
    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"
    
    # Topic modeling
    n_topics: int = 10
    n_top_words: int = 10
    min_word_length: int = 3

# Global configuration
config = ModelConfig()

class ModelBase:
    """Base class for ML models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._is_ready = False
        
    def is_ready(self) -> bool:
        """Check if model is loaded and ready"""
        return self._is_ready
    
    def _ensure_loaded(self):
        """Ensure model is loaded before use"""
        if not self._is_ready:
            raise RuntimeError(f"Model {self.model_name} is not loaded. Call load() first.")

class SentimentAnalyzer(ModelBase):
    """
    Sentiment Analysis using DistilBERT
    Returns sentiment polarity and confidence scores
    """
    
    def __init__(self):
        super().__init__(config.sentiment_model)
        self.load()
    
    def load(self):
        """Load the sentiment analysis model"""
        try:
            logger.info(f"Loading sentiment model: {self.model_name}")
            
            # Create pipeline with explicit model loading
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=0 if config.device == "cuda" else -1,
                model_kwargs={"cache_dir": config.model_cache_dir}
            )
            
            # Test the model
            test_result = self.pipeline("Test sentence")
            logger.info(f"Sentiment model loaded successfully. Test: {test_result}")
            
            self._is_ready = True
            
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            self._is_ready = False
            raise
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with sentiment label, score, and confidence
        """
        self._ensure_loaded()
        
        try:
            # Truncate text if too long
            if len(text) > config.max_length * 4:  # Rough character estimate
                text = text[:config.max_length * 4]
            
            # Get prediction
            result = self.pipeline(text, truncation=True, max_length=config.max_length)[0]
            
            # Convert to normalized score (-1 to 1)
            if result['label'] == 'POSITIVE':
                score = result['score']
            else:  # NEGATIVE
                score = -result['score']
            
            return {
                'label': result['label'],
                'score': score,
                'confidence': result['score'],
                'raw_scores': {
                    'positive': result['score'] if result['label'] == 'POSITIVE' else 1 - result['score'],
                    'negative': result['score'] if result['label'] == 'NEGATIVE' else 1 - result['score']
                }
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            # Return neutral sentiment on error
            return {
                'label': 'NEUTRAL',
                'score': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for multiple texts efficiently
        
        Args:
            texts: List of input texts
            
        Returns:
            List of sentiment dictionaries
        """
        self._ensure_loaded()
        
        if not texts:
            return []
        
        try:
            # Process in batches
            results = []
            batch_size = config.batch_size
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Truncate long texts
                batch = [t[:config.max_length * 4] if len(t) > config.max_length * 4 else t 
                        for t in batch]
                
                # Get predictions
                batch_results = self.pipeline(
                    batch,
                    truncation=True,
                    max_length=config.max_length,
                    padding=True
                )
                
                # Process results
                for result in batch_results:
                    if result['label'] == 'POSITIVE':
                        score = result['score']
                    else:  # NEGATIVE
                        score = -result['score']
                    
                    results.append({
                        'label': result['label'],
                        'score': score,
                        'confidence': result['score'],
                        'raw_scores': {
                            'positive': result['score'] if result['label'] == 'POSITIVE' else 1 - result['score'],
                            'negative': result['score'] if result['label'] == 'NEGATIVE' else 1 - result['score']
                        }
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch sentiment analysis: {e}")
            # Return neutral sentiments on error
            return [self.analyze(text) for text in texts]

class EmotionDetector(ModelBase):
    """
    Emotion Detection using DistilRoBERTa
    Detects multiple emotions: joy, sadness, anger, fear, surprise, disgust
    """
    
    def __init__(self):
        super().__init__(config.emotion_model)
        self.emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
        self.load()
    
    def load(self):
        """Load the emotion detection model"""
        try:
            logger.info(f"Loading emotion model: {self.model_name}")
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=0 if config.device == "cuda" else -1,
                model_kwargs={"cache_dir": config.model_cache_dir},
                top_k=None  # Return all scores
            )
            
            # Test the model
            test_result = self.pipeline("I am happy")
            logger.info(f"Emotion model loaded successfully. Test: {test_result[0][:2]}")
            
            self._is_ready = True
            
        except Exception as e:
            logger.error(f"Failed to load emotion model: {e}")
            self._is_ready = False
            raise
    
    def detect(self, text: str) -> Dict[str, float]:
        """
        Detect emotions in a single text
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with emotion scores
        """
        self._ensure_loaded()
        
        try:
            # Truncate text if too long
            if len(text) > config.max_length * 4:
                text = text[:config.max_length * 4]
            
            # Get predictions
            results = self.pipeline(text, truncation=True, max_length=config.max_length)
            
            # Convert to emotion dictionary
            emotions = {}
            for item in results[0]:  # First (and only) result
                emotions[item['label'].lower()] = item['score']
            
            # Ensure all emotions are present
            for emotion in self.emotion_labels:
                if emotion not in emotions:
                    emotions[emotion] = 0.0
            
            # Normalize scores to sum to 1
            total = sum(emotions.values())
            if total > 0:
                emotions = {k: v / total for k, v in emotions.items()}
            
            return emotions
            
        except Exception as e:
            logger.error(f"Error in emotion detection: {e}")
            # Return equal distribution on error
            return {emotion: 1.0 / len(self.emotion_labels) 
                   for emotion in self.emotion_labels}
    
    def detect_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Detect emotions for multiple texts efficiently
        
        Args:
            texts: List of input texts
            
        Returns:
            List of emotion dictionaries
        """
        self._ensure_loaded()
        
        if not texts:
            return []
        
        try:
            results = []
            batch_size = config.batch_size
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Truncate long texts
                batch = [t[:config.max_length * 4] if len(t) > config.max_length * 4 else t 
                        for t in batch]
                
                # Get predictions
                batch_results = self.pipeline(
                    batch,
                    truncation=True,
                    max_length=config.max_length,
                    padding=True
                )
                
                # Process results
                for result in batch_results:
                    emotions = {}
                    for item in result:
                        emotions[item['label'].lower()] = item['score']
                    
                    # Ensure all emotions are present
                    for emotion in self.emotion_labels:
                        if emotion not in emotions:
                            emotions[emotion] = 0.0
                    
                    # Normalize scores
                    total = sum(emotions.values())
                    if total > 0:
                        emotions = {k: v / total for k, v in emotions.items()}
                    
                    results.append(emotions)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch emotion detection: {e}")
            # Fallback to individual processing
            return [self.detect(text) for text in texts]

class TopicModeler:
    """
    Topic Modeling using Latent Dirichlet Allocation (LDA)
    Extracts main topics from a collection of texts
    """
    
    def __init__(self):
        self.vectorizer = None
        self.lda_model = None
        self._is_ready = False
        self._setup_nltk()
    
    def _setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
        self._is_ready = True
    
    def is_ready(self) -> bool:
        """Check if model is ready"""
        return self._is_ready
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for topic modeling
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [token for token in tokens 
                 if token.isalpha() 
                 and len(token) >= config.min_word_length 
                 and token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def extract_topics(self, texts: List[str], num_topics: int = 5) -> List[Dict[str, Any]]:
        """
        Extract topics from a collection of texts
        
        Args:
            texts: List of input texts
            num_topics: Number of topics to extract
            
        Returns:
            List of topics with keywords and scores
        """
        if not texts or len(texts) < 3:
            return []
        
        try:
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Remove empty texts
            processed_texts = [t for t in processed_texts if t.strip()]
            
            if len(processed_texts) < 3:
                return []
            
            # Create TF-IDF features
            self.vectorizer = TfidfVectorizer(
                max_features=100,
                min_df=2,
                max_df=0.8,
                ngram_range=(1, 2)
            )
            
            doc_term_matrix = self.vectorizer.fit_transform(processed_texts)
            
            # Fit LDA model
            self.lda_model = LatentDirichletAllocation(
                n_components=min(num_topics, len(processed_texts)),
                random_state=42,
                max_iter=10
            )
            
            self.lda_model.fit(doc_term_matrix)
            
            # Extract topics
            feature_names = self.vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(self.lda_model.components_):
                # Get top words for this topic
                top_word_indices = topic.argsort()[-config.n_top_words:][::-1]
                top_words = [feature_names[i] for i in top_word_indices]
                top_scores = [topic[i] for i in top_word_indices]
                
                # Calculate topic coherence (simplified)
                coherence = np.mean(top_scores) / np.std(top_scores) if np.std(top_scores) > 0 else 0
                
                topics.append({
                    'id': topic_idx,
                    'keywords': top_words,
                    'keyword_scores': [float(s) for s in top_scores],
                    'coherence': float(coherence),
                    'top_terms': [
                        {'term': word, 'weight': float(score)} 
                        for word, score in zip(top_words[:5], top_scores[:5])
                    ]
                })
            
            # Sort by coherence
            topics.sort(key=lambda x: x['coherence'], reverse=True)
            
            return topics
            
        except Exception as e:
            logger.error(f"Error in topic extraction: {e}")
            return []
    
    def get_document_topics(self, text: str) -> List[Tuple[int, float]]:
        """
        Get topic distribution for a single document
        
        Args:
            text: Input text
            
        Returns:
            List of (topic_id, probability) tuples
        """
        if not self.vectorizer or not self.lda_model:
            return []
        
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Transform to features
            doc_features = self.vectorizer.transform([processed_text])
            
            # Get topic distribution
            topic_dist = self.lda_model.transform(doc_features)[0]
            
            # Return sorted topics
            topics = [(i, float(prob)) for i, prob in enumerate(topic_dist)]
            topics.sort(key=lambda x: x[1], reverse=True)
            
            return topics
            
        except Exception as e:
            logger.error(f"Error getting document topics: {e}")
            return []

class ModelManager:
    """
    Manages all ML models and provides utility functions
    """
    
    def __init__(self):
        self.sentiment_analyzer = None
        self.emotion_detector = None
        self.topic_modeler = None
        self.executor = ThreadPoolExecutor(max_workers=config.num_workers)
    
    def initialize_all(self):
        """Initialize all models"""
        logger.info("Initializing all models...")
        
        try:
            self.sentiment_analyzer = SentimentAnalyzer()
            self.emotion_detector = EmotionDetector()
            self.topic_modeler = TopicModeler()
            
            logger.info("All models initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            return False
    
    def analyze_comprehensive(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with all analysis results
        """
        results = {
            'sentiment': None,
            'emotions': None,
            'topics': None,
            'metadata': {
                'text_length': len(text),
                'word_count': len(text.split())
            }
        }
        
        # Run analyses in parallel
        futures = []
        
        if self.sentiment_analyzer and self.sentiment_analyzer.is_ready():
            futures.append(
                self.executor.submit(self.sentiment_analyzer.analyze, text)
            )
        
        if self.emotion_detector and self.emotion_detector.is_ready():
            futures.append(
                self.executor.submit(self.emotion_detector.detect, text)
            )
        
        # Get results
        if len(futures) >= 1:
            results['sentiment'] = futures[0].result()
        
        if len(futures) >= 2:
            results['emotions'] = futures[1].result()
        
        # Topic modeling requires multiple texts, so skip for single text
        
        return results
    
    def shutdown(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)

# Module-level functions
def setup_models():
    """
    Download and setup all required models
    Called once during initial setup
    """
    logger.info("Setting up ML models...")
    
    # Create model cache directory
    os.makedirs(config.model_cache_dir, exist_ok=True)
    
    # Test model loading
    try:
        logger.info("Testing sentiment model...")
        sentiment = SentimentAnalyzer()
        result = sentiment.analyze("This is a test sentence.")
        logger.info(f"Sentiment test result: {result}")
        
        logger.info("Testing emotion model...")
        emotion = EmotionDetector()
        result = emotion.detect("I am feeling great today!")
        logger.info(f"Emotion test result: {result}")
        
        logger.info("Testing topic model...")
        topic = TopicModeler()
        logger.info("Topic model ready")
        
        logger.info("All models setup successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Model setup failed: {e}")
        return False

def get_model_info() -> Dict[str, Any]:
    """Get information about available models"""
    return {
        'sentiment': {
            'name': config.sentiment_model,
            'type': 'sentiment-analysis',
            'labels': ['POSITIVE', 'NEGATIVE'],
            'device': config.device
        },
        'emotion': {
            'name': config.emotion_model,
            'type': 'emotion-detection',
            'labels': ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'],
            'device': config.device
        },
        'topic': {
            'name': 'LDA',
            'type': 'topic-modeling',
            'algorithm': 'Latent Dirichlet Allocation',
            'features': 'TF-IDF'
        }
    }

# Example usage and testing
if __name__ == "__main__":
    # Test the models
    logger.info("Running model tests...")
    
    # Test sentiment analysis
    sentiment = SentimentAnalyzer()
    test_texts = [
        "I love this amazing product!",
        "This is terrible and disappointing.",
        "It's okay, nothing special."
    ]
    
    for text in test_texts:
        result = sentiment.analyze(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result}")
    
    # Test batch processing
    batch_results = sentiment.analyze_batch(test_texts)
    print(f"\nBatch results: {batch_results}")
    
    # Test emotion detection
    emotion = EmotionDetector()
    emotion_result = emotion.detect("I am so excited and happy about this news!")
    print(f"\nEmotion detection: {emotion_result}")
    
    # Test topic modeling
    topic_modeler = TopicModeler()
    sample_texts = [
        "Machine learning and artificial intelligence are transforming technology.",
        "Deep learning models require large datasets for training.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret visual information.",
        "Neural networks are inspired by the human brain structure."
    ]
    
    topics = topic_modeler.extract_topics(sample_texts, num_topics=2)
    print(f"\nExtracted topics: {json.dumps(topics, indent=2)}")
    
    logger.info("Model tests completed!")
