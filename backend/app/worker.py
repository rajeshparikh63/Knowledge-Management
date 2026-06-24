"""
Celery Worker Configuration
Handles background task processing for document ingestion
"""
from celery import Celery
from app.settings import settings

# Build Redis URL with authentication if password is provided
def build_redis_url(db: int) -> str:
    """Build Redis URL with optional authentication"""
    if settings.REDIS_PASSWORD:
        # Include password in URL (format: redis://:password@host:port/db)
        return f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}/{db}"
    else:
        # No password (local development)
        return f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{db}"

# IMPORTANT — broker/queue isolation.
# Another project on this machine (AI-Agency) runs its OWN Celery worker against
# the SAME Redis using broker db0 / results db1 and the default 'celery' queue.
# When we shared those, that worker would grab some of OUR tasks, fail to find
# them in its registry, and silently discard them as "unregistered" — leaving
# documents frozen at 'initializing' forever (the random "stuck files" bug).
# Give Knowledge-Management its own Redis DBs (5/6) AND its own queue name so the
# two projects' workers never consume each other's messages.
celery_app = Celery(
    "soldieriq_worker",
    broker=build_redis_url(5),
    backend=build_redis_url(6),
)

# Import task modules to register them with Celery
# This must happen after celery_app is created
import tasks.ingestion_tasks  # noqa: E402, F401

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Our own queue name (the other project uses the default 'celery' queue).
    # The worker consumes its task_default_queue automatically, so no -Q flag
    # is needed; producers (the API server) publish here too since they import
    # this same celery_app.
    task_default_queue="km_ingestion",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    task_soft_time_limit=3000,  # 50 minutes soft limit
    task_acks_late=True,  # Acknowledge after task completes
    # Prefetch a few tasks per thread so all 8 threads stay busy (with
    # acks_late, unprocessed prefetched tasks are redelivered if the worker
    # dies, so this is safe).
    worker_prefetch_multiplier=2,
    # NOTE: worker_max_tasks_per_child was 1 ("restart worker after each task").
    # That was a band-aid from before the persistent background-thread event
    # loop existed — it recycled the worker after EVERY task, which killed
    # concurrency and orphaned the other reserved tasks (they got stuck at
    # 'initializing' forever). With the persistent loop, the worker can safely
    # live across tasks. Recycle every 200 tasks purely for memory hygiene —
    # high enough that a normal import batch never recycles mid-flight.
    worker_max_tasks_per_child=200,
    worker_pool_restarts=True,  # Enable pool restarts
)
