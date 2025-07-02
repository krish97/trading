# Trading System

A microservices-based trading system with sentiment analysis, backend API, and frontend dashboard. Each service runs independently and can be started/stopped separately.

## Architecture

```
trading/
├── backend/           # FastAPI backend service
├── frontend/          # React frontend dashboard  
└── sentiment-analysis/   # Google Vertex AI sentiment service
```

## Services Overview

### 1. Sentiment Analysis Service
- **Port**: 9000
- **Technology**: Python, Google Vertex AI, FastAPI
- **Purpose**: Analyzes market sentiment for stock symbols using AI

### 2. Backend Service  
- **Port**: 8000
- **Technology**: Python, FastAPI, Polygon.io API
- **Purpose**: Trading signals, technical analysis, stock data API

### 3. Frontend Service
- **Port**: 3000  
- **Technology**: React, Vite, Material-UI
- **Purpose**: Web dashboard for viewing trading signals and data

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Windows PowerShell (for .bat scripts)

### Starting Services

**Option 1: Start All Services**
```bash
# Start sentiment analysis first
cd sentiment-analysis
docker-compose up --build -d

# Start backend  
cd ../backend
docker-compose up --build -d

# Start frontend
cd ../frontend  
docker-compose up --build -d
```

**Option 2: Use Convenience Scripts (Windows)**
```bash
# Start sentiment analysis
cd sentiment-analysis
./start_sentiment.bat

# Start backend
cd ../backend  
./start.bat

# Start frontend
cd ../frontend
./start.bat
```

### Accessing Services
- **Frontend Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **Sentiment API**: http://localhost:9000

### Stopping Services
```bash
# Stop each service
cd sentiment-analysis && docker-compose down
cd ../backend && docker-compose down  
cd ../frontend && docker-compose down
```

Or use the stop scripts:
```bash
cd sentiment-analysis && ./stop_sentiment.bat
cd ../backend && ./stop.bat
cd ../frontend && ./stop.bat
```

## Service Dependencies

```
Frontend (3000) → Backend (8000) → Sentiment (9000)
```

- Frontend calls Backend API for trading data
- Backend calls Sentiment API for market sentiment analysis
- Services communicate via HTTP APIs on localhost

## Development

Each service is completely independent and can be:
- Developed separately
- Deployed independently  
- Scaled independently
- Modified without affecting others

### Service-Specific Documentation
- `backend/` - FastAPI backend with trading algorithms
- `frontend/` - React dashboard with real-time data  
- `sentiment-analysis/` - AI-powered sentiment analysis

## Configuration

Each service has its own configuration:
- `backend/.env` - API keys, service URLs
- `sentiment-analysis/.env` - Google Cloud credentials
- `frontend/` - Build-time environment variables

## API Endpoints

### Backend API (port 8000)
- `GET /api/health` - Health check
- `GET /api/price/{symbol}` - Stock price data
- `GET /api/sentiment/{symbol}` - Sentiment analysis  
- `GET /api/signal/{symbol}` - Trading signals

### Sentiment API (port 9000)  
- `GET /health` - Health check
- `POST /analyze` - Analyze sentiment for symbol

## Technology Stack

| Service | Language | Framework | Database |
|---------|----------|-----------|----------|
| Frontend | JavaScript | React + Vite | None |
| Backend | Python | FastAPI | None |  
| Sentiment | Python | FastAPI | None |

## Deployment

Each service can be deployed independently:
- Each has its own `Dockerfile` and `docker-compose.yml`
- Services communicate via HTTP APIs
- No shared databases or file systems
- Horizontal scaling possible for each service

## Troubleshooting

1. **Services won't start**: Check Docker is running
2. **API calls fail**: Ensure services started in correct order
3. **Frontend shows errors**: Verify backend is running on port 8000
4. **No sentiment data**: Check sentiment service on port 9000

## License

MIT License 