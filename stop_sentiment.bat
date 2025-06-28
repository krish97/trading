@echo off
echo Stopping Sentiment Analysis Service...
cd sentiment-analysis
docker-compose down
echo.
echo Sentiment Analysis Service stopped.
pause 