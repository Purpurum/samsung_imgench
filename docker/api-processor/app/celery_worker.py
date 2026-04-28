from celery import Celery
import os

redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")

celery_app = Celery(
    "sentinel_processor",
    broker=redis_url,
    backend=redis_url
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=7200,
    worker_prefetch_multiplier=1,
    imports=("app.tasks",),
    broker_connection_retry_on_startup=True
)

# celery_app.autodiscover_tasks(["app"])