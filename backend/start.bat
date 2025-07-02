@echo off
echo Starting Backend Service on port 8000...
docker-compose up --build -d
echo.
echo Backend Service is starting...
echo Check status: docker-compose ps
echo View logs: docker-compose logs -f
echo Backend API: http://localhost:8000
echo.
echo Note: Make sure Sentiment Analysis Service is running on port 9000
pause 