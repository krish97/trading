# Spungus Trading System

A real-time trading dashboard with sentiment analysis and technical indicators based on the Spungus methodology.

## Architecture

The system consists of **two independent services**:

1. **Trading System** (Backend + Frontend)
   - FastAPI backend with Alpha Vantage integration (Port 8001)
   - React frontend with real-time charts (Port 8002)
   - Spungus EMA cloud and Fibonacci analysis
   - Option recommendations

2. **Sentiment Analysis Service** (Independent Microservice)
   - Real-time RSS news scraping (Port 9000)
   - Local LLM integration for sentiment scoring
   - Completely independent deployment
   - No fallbacks - fails fast on connection issues

## Quick Start

### Option 1: Start Both Services Together
```bash
# Start sentiment analysis service first
cd sentiment-analysis
docker-compose up --build -d

# Start main trading system
cd ..
docker-compose up --build -d
```

### Option 2: Use Convenience Scripts (Windows)
```bash
# Start sentiment analysis service
start_sentiment.bat

# Start main trading system
start_trading.bat

# Stop services
stop_sentiment.bat
stop_trading.bat
```

### Option 3: Independent Operation
```bash
# Run only sentiment analysis
cd sentiment-analysis
docker-compose up --build -d

# Run only trading system (will work without sentiment)
cd ..
docker-compose up --build -d
```

## Services

### Trading System
- **Backend API**: http://localhost:8001
- **Frontend Dashboard**: http://localhost:8002
- **Backend**: FastAPI with Alpha Vantage
- **Frontend**: React with Material UI and Chart.js
- **Features**: Real-time stock data, Spungus analysis, option recommendations

### Sentiment Analysis Service
- **URL**: http://localhost:9000
- **API**: FastAPI with RSS scraping and LLM integration
- **Features**: Real-time news sentiment, local LLM processing
- **Documentation**: See `sentiment-analysis/README.md`

## Prerequisites

- Docker and Docker Compose
- Alpha Vantage API key (optional, uses demo key by default)
- Local LLM server (e.g., LM Studio on port 1234) for sentiment analysis
- Internet access for RSS feeds

## Configuration

### Environment Variables
Create `.env` file in the root directory:
```env
ALPHA_VANTAGE_API_KEY=your_api_key_here
```

### Sentiment Service Configuration
See `sentiment-analysis/env.example` for sentiment service configuration.

## API Endpoints

### Trading System Backend (Port 8001)
- `GET /` - Backend health check
- `GET /api/price/{symbol}` - Stock price data
- `GET /api/signal/{symbol}` - Trading signals with sentiment
- `GET /api/health` - Health check

### Trading System Frontend (Port 8002)
- `GET /` - Main dashboard

### Sentiment Service (Port 9000)
- `GET /health` - Health check
- `POST /analyze` - Analyze sentiment for a symbol
- `POST /test-llm` - Test LLM integration
- `GET /` - Service info

## Features

### Spungus Methodology
- EMA cloud analysis (5/12 vs 34/50)
- Fibonacci retracement levels
- Trend identification and strength
- Option play recommendations

### Real-time Data
- Live stock prices from Alpha Vantage
- Real-time sentiment analysis from news
- Combined trading signals
- Beautiful React dashboard

### Independent Services
- Sentiment analysis can run independently
- Trading system works without sentiment (fallback)
- Easy to scale and maintain
- Separate logging and monitoring

## Troubleshooting

### Sentiment Service Issues
- Check `sentiment-analysis/README.md`
- Verify local LLM is running on port 1234
- Check RSS feed connectivity
- View logs: `cd sentiment-analysis && docker-compose logs -f`

### Trading System Issues
- Check Alpha Vantage API key
- Verify sentiment service is running on port 9000
- View logs: `docker-compose logs -f`

### General Issues
- Ensure Docker Desktop is running
- Check port availability (8001, 8002, 9000)
- Verify internet connectivity

## Development

### Running Services Independently
```bash
# Sentiment service only
cd sentiment-analysis
python sentiment_service.py

# Trading system only
cd backend
python backend.py
```

### Building Images
```bash
# Build sentiment service
cd sentiment-analysis
docker build -t sentiment-analysis .

# Build trading system
docker build -t trading-system .
```

## License

This project implements the Spungus trading methodology for educational purposes.