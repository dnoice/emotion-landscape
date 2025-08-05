# Emotion-Driven Data Landscape

**A unique ML-powered web experience that visualizes the emotional pulse of the internet**

🔗 **Repository**: [https://github.com/dnoice/emotion-landscape](https://github.com/dnoice/emotion-landscape)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌟 Project Overview

This project creates an interactive web application that:
- Fetches data from multiple public sources (news, social media, quotes)
- Analyzes sentiment and emotions using state-of-the-art ML models
- Creates stunning, real-time visualizations that morph based on collective emotions
- Provides an immersive, modern web experience with smooth animations

## 🏗️ Architecture

```
emotion-landscape/
├── backend/
│   ├── app.py                 # Flask API server (main entry point)
│   ├── ml_models.py           # ML model implementations
│   ├── data_processor.py      # Data fetching and processing
│   ├── cache_manager.py       # Intelligent caching system
│   ├── config.py              # Configuration management
│   ├── requirements.txt       # Python dependencies
│   └── models/               # Saved ML models directory
├── frontend/
│   ├── index.html            # Main HTML page
│   ├── css/
│   │   └── styles.css        # Modern, responsive styling
│   ├── js/
│   │   ├── app.js           # Main application controller
│   │   ├── visualizer.js    # D3.js/Three.js visualizations
│   │   ├── api.js           # API communication layer
│   │   └── particles.js     # Particle system for emotions
│   └── assets/              
│       ├── fonts/           # Custom fonts
│       └── sounds/          # Ambient sounds (optional)
├── data/
│   ├── cache/               # Cached API responses
│   └── datasets/            # Static datasets
├── notebooks/
│   ├── model_exploration.ipynb   # ML model testing
│   └── data_analysis.ipynb       # Dataset analysis
├── tests/
│   ├── test_api.py
│   └── test_models.py
├── docker-compose.yml       # Container orchestration
├── Dockerfile               # Backend container
├── .env.example            # Environment variables template
├── .gitignore
└── README.md               # This file
```

## 🚀 Features

### Machine Learning
- **Sentiment Analysis**: Using DistilBERT fine-tuned on SST-2
- **Emotion Detection**: Multi-class emotion classification (joy, sadness, anger, fear, surprise, disgust)
- **Topic Modeling**: LDA for discovering trending themes
- **Real-time Processing**: Stream processing for live data

### Data Sources
- **News**: RSS feeds from major news outlets
- **Reddit**: Public API for trending posts
- **Twitter**: Academic API for public tweets (optional)
- **Quotes**: Inspirational and philosophical quotes database
- **Wikipedia**: Current events and trending articles

### Visualizations
- **Emotion Particles**: Each data point as a particle with color/size based on emotion
- **Sentiment Waves**: Flowing waves representing sentiment trends
- **Word Clouds**: Dynamic word clouds of trending topics
- **Network Graphs**: Connections between emotions and topics
- **Time Series**: Historical emotion trends

## 🛠️ Tech Stack

### Backend
- **Flask**: Lightweight Python web framework
- **Transformers**: Hugging Face models for NLP
- **Pandas/NumPy**: Data processing
- **Redis**: Caching layer (optional)
- **Celery**: Background task processing (optional)

### Frontend
- **Vanilla JS**: No framework dependencies for maximum performance
- **D3.js**: Data visualizations
- **Three.js**: 3D particle effects
- **Web Audio API**: Ambient soundscapes
- **CSS Grid/Flexbox**: Modern responsive layout

## 📦 Installation

### Prerequisites
- Python 3.8+
- Node.js 16+ (for development tools)
- Git

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/emotion-landscape.git
cd emotion-landscape
```

2. **Set up Python environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
cd backend
pip install -r requirements.txt
```

3. **Download ML models** (first run will auto-download)
```bash
python -c "from ml_models import setup_models; setup_models()"
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys (optional)
```

5. **Run the backend**
```bash
python app.py
```

6. **Open the frontend**
```bash
# In a new terminal
cd frontend
# For development, use a local server
python -m http.server 8000
# Or use Node.js
npx http-server -p 8000
```

7. **Visit** http://localhost:8000

## 🔧 Configuration

### Environment Variables
```bash
# .env file
FLASK_ENV=development
FLASK_DEBUG=True
API_PORT=5000

# Optional API Keys
NEWS_API_KEY=your_key_here
TWITTER_BEARER_TOKEN=your_token_here

# ML Model Settings
MODEL_CACHE_DIR=./models
MAX_TEXT_LENGTH=512
BATCH_SIZE=32

# Caching
CACHE_TTL=3600  # 1 hour
ENABLE_REDIS=False
```

## 📊 API Endpoints

### GET /api/analyze
Fetch and analyze data from specified source
```json
// Request
GET /api/analyze?source=news&limit=50

// Response
{
  "status": "success",
  "data": [
    {
      "id": "123",
      "text": "Sample text...",
      "sentiment": {
        "label": "POSITIVE",
        "score": 0.85
      },
      "emotions": {
        "joy": 0.7,
        "surprise": 0.2,
        ...
      }
    }
  ],
  "summary": {
    "average_sentiment": 0.65,
    "dominant_emotion": "joy"
  }
}
```

### GET /api/sources
Get available data sources

### POST /api/analyze/text
Analyze custom text

### GET /api/trends
Get emotion trends over time

## 🎨 Customization

### Adding New Data Sources
1. Create a new fetcher in `data_processor.py`
2. Add source to `get_sources()` endpoint
3. Update frontend source selector

### Creating New Visualizations
1. Add visualization module to `js/visualizations/`
2. Register in `visualizer.js`
3. Add controls to UI

### Training Custom Models
See `notebooks/model_training.ipynb` for examples

## 🧪 Testing

```bash
# Run backend tests
cd backend
pytest tests/

# Run frontend tests (if implemented)
cd frontend
npm test
```

## 🚢 Deployment

### Docker Deployment
```bash
docker-compose up --build
```

### Manual Deployment
See `docs/deployment.md` for detailed instructions

## 📈 Performance Considerations

- Models are loaded once at startup
- API responses are cached for 1 hour
- Frontend uses requestAnimationFrame for smooth animations
- Batch processing for multiple texts
- WebSocket support for real-time updates (optional)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

Repository: https://github.com/dnoice/emotion-landscape

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- Hugging Face for pre-trained models
- D3.js community for visualization examples
- Public API providers for data access
