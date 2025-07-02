FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*
    
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY google_genai_client.py .
COPY sentiment_service.py .

EXPOSE 8001

CMD ["uvicorn", "sentiment_service:app", "--host", "0.0.0.0", "--port", "8001"] 