# Plain python base (not the tiangolo FastAPI image) — needed to control the
# Python version torch/faiss/lightgbm wheels are compatible with.
FROM python:3.11-slim

WORKDIR /app

# lightgbm's compiled wheel dynamically links against libgomp (OpenMP) at
# import time — not present on -slim Debian images by default. Without this,
# the container builds fine but crashes on startup with
# "OSError: libgomp.so.1: cannot open shared object file".
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch build — the default PyPI wheel pulls CUDA and is ~2GB+
# larger for no benefit, since this service only ever runs a small MLP
# forward pass per request (see model_loader.py), not training.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN sed -i 's/\r$//' /app/prestart.sh && chmod +x /app/prestart.sh

EXPOSE 8000

CMD ["bash", "-c", "./prestart.sh && uvicorn main:app --host 0.0.0.0 --port 8000"]
