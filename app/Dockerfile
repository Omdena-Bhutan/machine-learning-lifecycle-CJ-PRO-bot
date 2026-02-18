FROM python:3.10-slim

WORKDIR /app

COPY app/requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app code (includes api.py and models folder if present)
COPY app/ .

EXPOSE 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "api:app"]
