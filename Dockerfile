FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY fraude/ fraude/
COPY api.py .
COPY serving/ serving/
COPY web/ web/

ENV MODEL=b SERVING_DIR=serving LOG_DIR=/tmp/logs
EXPOSE 8000
# Cloud Run y otros hosts definen PORT; en local usa 8000.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
