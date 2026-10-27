FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Train inside the image, so the model always matches the installed library versions
COPY churn/ churn/
COPY data/telco_churn.csv data/
COPY train.py .
RUN python train.py

COPY app/ app/

EXPOSE 8000
# Hosting platforms such as Render pass the port in $PORT; locally it defaults to 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
