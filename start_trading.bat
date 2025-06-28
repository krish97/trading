@echo off
echo Starting Trading System (Backend: 8001, Frontend: 8002)...
docker-compose up --build -d
echo.
echo Trading System is starting...
echo Check status: docker-compose ps
echo View logs: docker-compose logs -f
echo Backend API: http://localhost:8001
echo Frontend Dashboard: http://localhost:8002
echo.
echo Note: Make sure Sentiment Analysis Service is running on port 9000
echo Start sentiment service: start_sentiment.bat
pause 