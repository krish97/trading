@echo off
echo Starting Sentiment Analysis Service...
cd sentiment-analysis
docker-compose up --build -d
echo.
echo Sentiment Analysis Service is starting...
echo Check status: docker-compose ps
echo View logs: docker-compose logs -f
echo Health check: http://localhost:9000/health
pause 