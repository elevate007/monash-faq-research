FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/cache/huggingface FAQ_BACKEND=extractive
WORKDIR /app
COPY requirements-api.txt requirements-inference.txt ./
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-inference.txt
COPY service ./service
COPY src/__init__.py src/retrieval.py ./src/
COPY data ./data
COPY results/comparison/calibration_result.json ./results/comparison/calibration_result.json
RUN useradd --uid 10001 --create-home appuser && mkdir -p /cache && chown appuser:appuser /cache
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz')"
CMD ["uvicorn","service.app:app","--host","0.0.0.0","--port","8000","--workers","1","--no-access-log"]
