# Use an official Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (better Docker caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start the API server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]