# Sentiment Analysis Service

A standalone microservice for real-time financial news sentiment analysis using local LLM integration.

## Features

- **Real-time RSS scraping** from multiple financial news sources
- **Local LLM integration** for advanced sentiment analysis
- **RESTful API** with FastAPI
- **Docker containerization** for easy deployment
- **Health monitoring** and logging
- **Real data only** - no fallbacks, no test data, fails fast on issues

## Prerequisites

- Docker and Docker Compose
- Local LLM server running (e.g., LM Studio on port 1234)
- Internet access for RSS feeds

## Quick Start

1. **Clone and navigate to the sentiment service directory:**
   ```bash
   cd sentiment-analysis
   ```

2. **Copy environment file:**
   ```bash
   cp env.example .env
   ```

3. **Start the service:**
   ```bash
   docker-compose up --build -d
   ```

4. **Check health:**
   ```bash
   curl http://localhost:9000/health
   ```

## API Endpoints

### Health Check
```bash
GET /health
```

### Analyze Sentiment
```bash
POST /analyze
Content-Type: application/json

{
  "symbol": "NVDA",
  "max_articles": 10
}
```

### Test LLM with Real Data
```bash
POST /test-llm
```

### Service Info
```bash
GET /
```

## Configuration

Edit `.env` file to customize:

- `LOCAL_LLM_URL`: URL of your local LLM server
- `LOCAL_LLM_MODEL`: Model name for LLM
- `SERVICE_PORT`: Port for the service (default: 8001)
- `RSS_TIMEOUT`: Timeout for RSS feed requests
- `MAX_ARTICLES_PER_SOURCE`: Max articles per news source

## News Sources

The service scrapes from:
- Reuters Business
- Yahoo Finance
- MarketWatch
- Seeking Alpha
- Investing.com
- Benzinga
- FinViz
- StockTwits

## Logs

View logs:
```bash
docker-compose logs -f sentiment-analysis
```

Logs are also mounted to `./logs/` directory.

## Independent Operation

This service can run completely independently:

```bash
# Start only sentiment service
docker-compose up -d

# Stop sentiment service
docker-compose down

# Rebuild and restart
docker-compose up --build -d

# View logs
docker-compose logs -f
```

## Integration with Main Trading System

When running with the main trading system, the sentiment service URL should be:
```
http://localhost:9000
```

## Troubleshooting

1. **Service won't start:**
   - Check Docker is running
   - Verify port 9000 is available
   - Check logs: `docker-compose logs sentiment-analysis`

2. **LLM connection fails:**
   - Ensure local LLM server is running
   - Check `LOCAL_LLM_URL` in `.env`
   - Verify network connectivity
   - Service will fail completely if LLM is not available

3. **RSS feeds blocked:**
   - Check internet connectivity
   - Verify RSS feed URLs are accessible
   - Service will fail fast with clear error messages
   - No fallback to simulated data

4. **No test data:**
   - All endpoints use real RSS data only
   - `/test-llm` scrapes real articles for testing
   - No simulated or fallback data available

## Development

To run in development mode:
```bash
# Install dependencies
pip install -r requirements.txt

# Run directly
python sentiment_service.py
```

## License

This service is part of the Spungus Trading System. 