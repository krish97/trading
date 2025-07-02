# Sentiment Analysis Service

A standalone microservice for performing advanced financial sentiment analysis using Google's Vertex AI.

## Features

- **Advanced AI Analysis**: Utilizes the `gemini-2.5-pro` model to perform research and analysis.
- **Agentic Workflow**: The model acts as a financial analyst, using its own Google Search tool to find and process the latest information about a stock symbol.
- **Rich, Structured Output**: Returns a detailed JSON object containing not just a sentiment score, but also key bullish/bearish factors, an investment recommendation, and more.
- **Cost-Effective Batch API**: Uses the Vertex AI Batch API, which provides a 50% cost reduction compared to standard calls.
- **RESTful API**: Built with FastAPI for easy integration.
- **Docker Containerization**: Fully containerized for portability and simplified deployment.
- **Health Monitoring**: Includes a `/health` endpoint for monitoring service status.

## Prerequisites

- Docker and Docker Compose
- A Google Cloud Platform project with the Vertex AI API enabled.
- A service account key with the "Vertex AI User" role.

## Configuration

The service is configured via environment variables, typically set in the main `docker-compose.yml` file:

- `GCP_PROJECT_ID`: Your Google Cloud Project ID.
- `GCP_LOCATION`: The GCP region for your Vertex AI resources (e.g., `us-central1`).
- `GOOGLE_APPLICATION_CREDENTIALS`: The path inside the container to the service account key file (e.g., `/app/vertex-sa-key.json`).

## API Endpoints

### Health Check
- **Endpoint**: `GET /health`
- **Description**: Checks the service status and the LLM client initialization.

### Analyze a Single Symbol
- **Endpoint**: `POST /analyze`
- **Body**: `{ "symbol": "NVDA" }`
- **Description**: Triggers a full agentic analysis for a single stock symbol.

### Analyze a Batch of Symbols
- **Endpoint**: `POST /analyze-batch`
- **Body**: `{ "symbols": ["NVDA", "AAPL", "TSLA"] }`
- **Description**: Analyzes multiple stock symbols in a single, cost-effective batch request.

## Logs

View the service's logs using Docker Compose from the project's root directory:
```bash
docker-compose logs -f sentiment-analysis
```
Logs are also mounted to the `./sentiment-analysis/logs/` directory on the host.

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