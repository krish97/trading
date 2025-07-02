@echo off
echo Starting Frontend Service on port 3000...
docker-compose up --build -d
echo.
echo Frontend Service is starting...
echo Check status: docker-compose ps
echo View logs: docker-compose logs -f
echo Frontend Dashboard: http://localhost:3000
echo.
echo Note: Make sure Backend Service is running on port 8000
pause 