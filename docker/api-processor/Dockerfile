FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONPATH=/app/src:/usr/local/snap/snappy:${PYTHONPATH}

RUN python3 -m venv $VIRTUAL_ENV && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --only-binary=:all: \
        "fastapi==0.109.2" \
		"pillow" \
        "uvicorn==0.27.1" \
        "celery==5.3.6" \
        "redis==5.0.1" \
        "boto3==1.34.34" \
        "shapely>=2.0.3" \
        "python-multipart==0.0.6"

WORKDIR /app
COPY app/ ./app/

EXPOSE 8000