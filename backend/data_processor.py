#!/usr/bin/env python3
"""
File: backend/data_processor.py
Description: Data fetching and processing from various public sources
Author: Dennis Smaltz
Acknowledgement: Claude Opus 4
Created: 2024
Python Version: 3.8+

This module handles:
- Fetching data from public APIs (Reddit, Wikipedia, quotes)
- RSS feed parsing for news
- Data cleaning and normalization
- Rate limiting and error handling
"""

import os
import json
import time
import random
import logging
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from urllib.parse import urlparse, parse_qs
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import requests
import feedparser
from bs4 import BeautifulSoup
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = defaultdict(float)
    
    def wait_if_needed(self, source: str):
        """Wait if necessary to respect rate limits"""
        now = time.time()
        elapsed = now - self.last_call[source]
        
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        
        self.last_call[source] = time.time()

class DataProcessor:
    """
    Fetches and processes data from various public sources
    """
    
    def __init__(self, cache_dir: str = "../data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Rate limiter
        self.rate_limiter = RateLimiter(calls_per_minute=30)
        
        # User agent for requests
        self.headers = {
            'User-Agent': 'EmotionLandscape/1.0 (https://github.com/dnoice/emotion-landscape)'
        }
        
        # Thread pool for concurrent fetching
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize data sources
        self._init_sources()
    
    def _init_sources(self):
        """Initialize data source configurations"""
        self.news_sources = [
            {
                'name': 'BBC News',
                'url': 'http://feeds.bbci.co.uk/news/rss.xml',
                'type': 'rss'
            },
            {
                'name': 'Reuters',
                'url': 'https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best',
                'type': 'rss'
            },
            {
                'name': 'The Guardian',
                'url': 'https://www.theguardian.com/world/rss',
                'type': 'rss'
            },
            {
                'name': 'TechCrunch',
                'url': 'https://techcrunch.com/feed/',
                'type': 'rss'
            }
        ]
        
        self.quote_sources = [
            {
                'name': 'ZenQuotes',
                'url': 'https://zenquotes.io/api/quotes',
                'type': 'api'
            }
        ]
    
    def fetch_data(self, source: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch data from specified source
        
        Args:
            source: Data source identifier
            limit: Maximum number of items to fetch
            
        Returns:
            List of processed data items
        """
        logger.info(f"Fetching data from source: {source} (limit: {limit})")
        
        try:
            if source == 'news':
                return self.fetch_news(limit)
            elif source == 'reddit':
                return self.fetch_reddit(limit)
            elif source == 'quotes':
                return self.fetch_quotes(limit)
            elif source == 'wikipedia':
                return self.fetch_wikipedia(limit)
            else:
                logger.warning(f"Unknown source: {source}")
                return self._get_fallback_data(source, limit)
                
        except Exception as e:
            logger.error(f"Error fetching data from {source}: {e}")
            return self._get_fallback_data(source, limit)
    
    def fetch_news(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch news from RSS feeds"""
        all_news = []
        
        # Fetch from multiple sources concurrently
        futures = []
        for source in self.news_sources:
            future = self.executor.submit(self._fetch_rss_feed, source['url'], source['name'])
            futures.append(future)
        
        # Collect results
        for future in as_completed(futures):
            try:
                news_items = future.result()
                all_news.extend(news_items)
            except Exception as e:
                logger.error(f"Error fetching news: {e}")
        
        # Sort by timestamp and limit
        all_news.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return all_news[:limit]
    
    def _fetch_rss_feed(self, url: str, source_name: str) -> List[Dict[str, Any]]:
        """Fetch and parse RSS feed"""
        try:
            self.rate_limiter.wait_if_needed('rss')
            
            # Parse feed
            feed = feedparser.parse(url)
            
            if feed.bozo:
                logger.warning(f"Feed parsing error for {source_name}: {feed.bozo_exception}")
                return []
            
            items = []
            for entry in feed.entries[:20]:  # Limit per feed
                # Extract text content
                text = entry.get('title', '')
                if 'summary' in entry:
                    text += ' ' + self._clean_html(entry['summary'])
                elif 'description' in entry:
                    text += ' ' + self._clean_html(entry['description'])
                
                # Create item
                item = {
                    'id': hashlib.md5(entry.get('link', str(random.random())).encode()).hexdigest()[:12],
                    'text': text.strip(),
                    'title': entry.get('title', 'No title'),
                    'url': entry.get('link', ''),
                    'source': source_name,
                    'timestamp': self._parse_timestamp(entry.get('published', entry.get('updated', ''))),
                    'author': entry.get('author', ''),
                    'tags': [tag['term'] for tag in entry.get('tags', [])][:5]
                }
                
                if item['text']:  # Only add if has content
                    items.append(item)
            
            return items
            
        except Exception as e:
            logger.error(f"Error fetching RSS feed {url}: {e}")
            return []
    
    def fetch_reddit(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch posts from Reddit's public API"""
        try:
            self.rate_limiter.wait_if_needed('reddit')
            
            # Popular subreddits for diverse content
            subreddits = ['all', 'worldnews', 'technology', 'science', 'AskReddit']
            all_posts = []
            
            posts_per_sub = max(10, limit // len(subreddits))
            
            for subreddit in subreddits:
                try:
                    url = f'https://www.reddit.com/r/{subreddit}/hot.json'
                    params = {'limit': posts_per_sub}
                    
                    response = requests.get(url, headers=self.headers, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        for post in data['data']['children']:
                            post_data = post['data']
                            
                            # Combine title and text
                            text = post_data['title']
                            if post_data.get('selftext'):
                                text += ' ' + post_data['selftext']
                            
                            # Skip if too short
                            if len(text.split()) < 5:
                                continue
                            
                            item = {
                                'id': post_data['id'],
                                'text': text[:1000],  # Limit length
                                'title': post_data['title'],
                                'url': f"https://reddit.com{post_data['permalink']}",
                                'source': f"r/{post_data['subreddit']}",
                                'timestamp': datetime.fromtimestamp(post_data['created_utc']).isoformat(),
                                'author': post_data.get('author', '[deleted]'),
                                'score': post_data.get('score', 0),
                                'num_comments': post_data.get('num_comments', 0),
                                'tags': [post_data['subreddit'], post_data.get('link_flair_text', '')],
                                'nsfw': post_data.get('over_18', False)
                            }
                            
                            # Filter out NSFW content
                            if not item['nsfw']:
                                all_posts.append(item)
                    
                    time.sleep(0.5)  # Be nice to Reddit's servers
                    
                except Exception as e:
                    logger.error(f"Error fetching from r/{subreddit}: {e}")
            
            # Sort by score and timestamp
            all_posts.sort(key=lambda x: (x['score'], x['timestamp']), reverse=True)
            
            return all_posts[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching Reddit data: {e}")
            return self._get_fallback_data('reddit', limit)
    
    def fetch_quotes(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch inspirational quotes"""
        quotes = []
        
        # Try ZenQuotes API first
        try:
            self.rate_limiter.wait_if_needed('quotes')
            
            url = 'https://zenquotes.io/api/quotes'
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for i, quote in enumerate(data[:limit]):
                    if quote['q'] != 'Too many requests. Obtain an auth key for unlimited access.':
                        item = {
                            'id': f"quote_{i}_{int(time.time())}",
                            'text': quote['q'],
                            'source': 'quotes',
                            'author': quote['a'],
                            'timestamp': datetime.utcnow().isoformat(),
                            'tags': ['inspirational', 'wisdom'],
                            'url': ''
                        }
                        quotes.append(item)
            
        except Exception as e:
            logger.error(f"Error fetching from ZenQuotes: {e}")
        
        # Fallback to static quotes if needed
        if len(quotes) < limit:
            quotes.extend(self._get_static_quotes(limit - len(quotes)))
        
        return quotes[:limit]
    
    def fetch_wikipedia(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch current events from Wikipedia"""
        try:
            self.rate_limiter.wait_if_needed('wikipedia')
            
            # Wikipedia API for current events
            base_url = 'https://en.wikipedia.org/api/rest_v1/page/summary/'
            
            # Get featured articles and current events
            events = []
            
            # Try to get "Portal:Current_events" content
            try:
                url = 'https://en.wikipedia.org/api/rest_v1/page/segments/Portal:Current_events'
                response = requests.get(url, headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    # This would need more complex parsing
                    # For now, use featured articles instead
                    pass
            except:
                pass
            
            # Get random featured articles
            url = 'https://en.wikipedia.org/api/rest_v1/page/random/summary'
            
            for i in range(min(limit, 20)):  # Wikipedia rate limits
                try:
                    response = requests.get(url, headers=self.headers, timeout=5)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data.get('extract'):
                            item = {
                                'id': f"wiki_{data.get('pageid', i)}",
                                'text': data['extract'],
                                'title': data.get('title', ''),
                                'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                                'source': 'Wikipedia',
                                'timestamp': datetime.utcnow().isoformat(),
                                'tags': ['encyclopedia', 'knowledge'],
                                'author': 'Wikipedia'
                            }
                            events.append(item)
                    
                    time.sleep(0.1)  # Respect rate limits
                    
                except Exception as e:
                    logger.error(f"Error fetching Wikipedia article: {e}")
            
            return events
            
        except Exception as e:
            logger.error(f"Error fetching Wikipedia data: {e}")
            return self._get_fallback_data('wikipedia', limit)
    
    def _clean_html(self, html: str) -> str:
        """Remove HTML tags from text"""
        if not html:
            return ''
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text(separator=' ', strip=True)
        except:
            # Fallback to regex if BeautifulSoup fails
            import re
            return re.sub('<[^<]+?>', '', html)
    
    def _parse_timestamp(self, timestamp_str: str) -> str:
        """Parse various timestamp formats to ISO format"""
        if not timestamp_str:
            return datetime.utcnow().isoformat()
        
        try:
            # Try parsing with feedparser's method
            parsed = feedparser._parse_date(timestamp_str)
            if parsed:
                return datetime(*parsed[:6]).isoformat()
        except:
            pass
        
        # Fallback to current time
        return datetime.utcnow().isoformat()
    
    def _get_static_quotes(self, count: int) -> List[Dict[str, Any]]:
        """Get static quotes as fallback"""
        static_quotes = [
            ("The only way to do great work is to love what you do.", "Steve Jobs"),
            ("In the middle of difficulty lies opportunity.", "Albert Einstein"),
            ("Success is not final, failure is not fatal: it is the courage to continue that counts.", "Winston Churchill"),
            ("The future belongs to those who believe in the beauty of their dreams.", "Eleanor Roosevelt"),
            ("It does not matter how slowly you go as long as you do not stop.", "Confucius"),
            ("Everything you've ever wanted is on the other side of fear.", "George Addair"),
            ("Believe you can and you're halfway there.", "Theodore Roosevelt"),
            ("The only impossible journey is the one you never begin.", "Tony Robbins"),
            ("Life is 10% what happens to you and 90% how you react to it.", "Charles R. Swindoll"),
            ("The way to get started is to quit talking and begin doing.", "Walt Disney"),
            ("Don't watch the clock; do what it does. Keep going.", "Sam Levenson"),
            ("The pessimist sees difficulty in every opportunity. The optimist sees opportunity in every difficulty.", "Winston Churchill"),
            ("You learn more from failure than from success. Don't let it stop you. Failure builds character.", "Unknown"),
            ("It's not whether you get knocked down, it's whether you get up.", "Vince Lombardi"),
            ("If you are working on something that you really care about, you don't have to be pushed.", "Steve Jobs"),
            ("People who are crazy enough to think they can change the world, are the ones who do.", "Rob Siltanen"),
            ("We may encounter many defeats but we must not be defeated.", "Maya Angelou"),
            ("Knowing is not enough; we must apply. Wishing is not enough; we must do.", "Johann Wolfgang Von Goethe"),
            ("Whether you think you can or think you can't, you're right.", "Henry Ford"),
            ("The only limit to our realization of tomorrow will be our doubts of today.", "Franklin D. Roosevelt"),
        ]
        
        quotes = []
        for i in range(min(count, len(static_quotes))):
            quote_text, author = static_quotes[i]
            quotes.append({
                'id': f"static_quote_{i}",
                'text': quote_text,
                'source': 'quotes',
                'author': author,
                'timestamp': datetime.utcnow().isoformat(),
                'tags': ['inspirational', 'wisdom', 'motivation'],
                'url': ''
            })
        
        return quotes
    
    def _get_fallback_data(self, source: str, limit: int) -> List[Dict[str, Any]]:
        """Generate fallback data when API fails"""
        logger.info(f"Using fallback data for {source}")
        
        fallback_templates = {
            'news': [
                "Scientists discover breakthrough in renewable energy technology",
                "Global climate summit reaches historic agreement",
                "New AI technology promises to revolutionize healthcare",
                "Economic indicators show signs of recovery",
                "Space exploration reaches new milestone",
                "Technology companies announce major collaboration",
                "Medical researchers make progress on disease treatment",
                "Environmental conservation efforts show positive results",
                "International cooperation leads to peaceful resolution",
                "Educational initiatives improve access worldwide"
            ],
            'reddit': [
                "What's the most interesting thing you learned recently?",
                "Scientists of Reddit, what's the coolest discovery in your field?",
                "What technology do you think will change the world in the next decade?",
                "What's a small act of kindness that made a big difference in your life?",
                "What's the most beautiful place you've ever visited?",
                "What book changed your perspective on life?",
                "What's the best advice you've ever received?",
                "What scientific fact blows your mind every time?",
                "What's a hobby that's improved your life?",
                "What positive change have you made recently?"
            ],
            'quotes': self._get_static_quotes(limit),
            'wikipedia': [
                "The history of human civilization spans thousands of years of development.",
                "Scientific method has been fundamental to human progress and understanding.",
                "Art and culture reflect the diversity of human experience across the globe.",
                "Technological advancement continues to reshape how we live and work.",
                "Natural ecosystems demonstrate incredible complexity and interdependence.",
                "Human psychology reveals fascinating insights into behavior and cognition.",
                "Mathematical principles underlie many aspects of the natural world.",
                "Literature captures the essence of human experience across cultures.",
                "Music has been a universal form of human expression throughout history.",
                "Philosophy explores fundamental questions about existence and meaning."
            ]
        }
        
        templates = fallback_templates.get(source, fallback_templates['news'])
        
        if source == 'quotes':
            return templates[:limit]
        
        data = []
        for i in range(min(limit, len(templates))):
            item = {
                'id': f"{source}_fallback_{i}",
                'text': templates[i % len(templates)],
                'source': f"{source} (sample)",
                'timestamp': (datetime.utcnow() - timedelta(minutes=i)).isoformat(),
                'author': 'Sample Data',
                'tags': [source, 'sample'],
                'url': ''
            }
            
            if source == 'reddit':
                item['score'] = random.randint(100, 10000)
                item['num_comments'] = random.randint(10, 1000)
            
            data.append(item)
        
        return data
    
    def search_by_keyword(self, keyword: str, sources: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search for content containing specific keyword across sources
        
        Args:
            keyword: Search keyword
            sources: List of sources to search (None = all)
            
        Returns:
            List of matching items
        """
        if sources is None:
            sources = ['news', 'reddit', 'quotes', 'wikipedia']
        
        all_results = []
        
        for source in sources:
            try:
                data = self.fetch_data(source, limit=100)
                
                # Filter by keyword
                keyword_lower = keyword.lower()
                matching = [
                    item for item in data
                    if keyword_lower in item.get('text', '').lower()
                    or keyword_lower in item.get('title', '').lower()
                ]
                
                all_results.extend(matching)
                
            except Exception as e:
                logger.error(f"Error searching {source}: {e}")
        
        # Sort by relevance (simple scoring)
        for item in all_results:
            text = (item.get('text', '') + ' ' + item.get('title', '')).lower()
            item['relevance_score'] = text.count(keyword.lower())
        
        all_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return all_results[:50]  # Limit results
    
    def get_trending_topics(self) -> List[str]:
        """Get current trending topics across all sources"""
        # This would ideally use more sophisticated trending algorithms
        # For now, extract common words from recent data
        
        all_text = []
        for source in ['news', 'reddit']:
            try:
                data = self.fetch_data(source, limit=50)
                for item in data:
                    all_text.append(item.get('text', ''))
            except:
                pass
        
        # Simple word frequency analysis
        from collections import Counter
        import re
        
        words = []
        for text in all_text:
            # Extract words (simple tokenization)
            text_words = re.findall(r'\b\w+\b', text.lower())
            # Filter common words and short words
            words.extend([w for w in text_words if len(w) > 4])
        
        # Get most common words
        word_counts = Counter(words)
        common_words = word_counts.most_common(20)
        
        return [word for word, count in common_words]
    
    def shutdown(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)

# Utility functions
def test_data_processor():
    """Test data processor functionality"""
    processor = DataProcessor()
    
    # Test each source
    sources = ['news', 'reddit', 'quotes', 'wikipedia']
    
    for source in sources:
        print(f"\n{'='*50}")
        print(f"Testing {source} source:")
        print('='*50)
        
        data = processor.fetch_data(source, limit=5)
        
        if data:
            print(f"Fetched {len(data)} items")
            for i, item in enumerate(data[:2]):  # Show first 2
                print(f"\nItem {i+1}:")
                print(f"  ID: {item.get('id')}")
                print(f"  Source: {item.get('source')}")
                print(f"  Text: {item.get('text', '')[:100]}...")
                print(f"  Timestamp: {item.get('timestamp')}")
        else:
            print("No data fetched")
    
    processor.shutdown()

if __name__ == "__main__":
    test_data_processor()
