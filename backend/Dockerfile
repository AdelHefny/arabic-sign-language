# Use Python 3.10 slim as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY backend/main.py .

# Copy the labels file
COPY labels.json .

# Copy the model file
COPY models/arabic_sign_video_model_large.onnx .

# Set environment variables
ENV MODEL_PATH=/app/arabic_sign_video_model_large.onnx
ENV LABELS_PATH=/app/labels.json

# Expose port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
