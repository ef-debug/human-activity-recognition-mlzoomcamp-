# Python environment
FROM python:3.14.2-slim

# Working directory inside the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the prediction service and trained model
COPY predict.py .
COPY model.bin .

# Application port
EXPOSE 9696

# Start the prediction service
CMD ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]