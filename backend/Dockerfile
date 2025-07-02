FROM python:3.11-slim

# Prevent buffering
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Ensure static directory exists even if empty
RUN mkdir -p /app/static

# Copy application code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"] 