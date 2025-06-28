# ---- Build React frontend ----
FROM node:20-alpine as frontend-build

WORKDIR /app/frontend-react

# Copy package files
COPY frontend-react/package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY frontend-react/ ./

# Build the React app
RUN npm run build

# ---- Backend with FastAPI ----
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/backend.py ./backend/backend.py

# Copy built frontend
COPY --from=frontend-build /app/frontend-react/dist ./frontend
COPY --from=frontend-build /app/frontend-react/package*.json ./frontend/

# Install serve for frontend
RUN npm install -g serve

# Create startup script
RUN echo '#!/bin/bash\n\
# Start backend in background\n\
uvicorn backend.backend:app --host 0.0.0.0 --port 8000 &\n\
# Start frontend\n\
serve -s frontend -l 3000\n\
wait' > /app/start.sh && chmod +x /app/start.sh

# Expose both ports
EXPOSE 8000 3000

# Start both services
CMD ["/app/start.sh"] 