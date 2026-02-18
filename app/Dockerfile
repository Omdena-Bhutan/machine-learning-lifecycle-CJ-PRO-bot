FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for Docker cache
COPY app/requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy app code
COPY app/ .

# Copy your saved baseline models
COPY app/models/ ./models/

EXPOSE 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "api:app"]
