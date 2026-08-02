from celery import Celery
from app.config import Config

# Initialize Celery app
celery_app = Celery("analysis_worker")

# Load configuration from our config.py
celery_app.conf.update(
    broker_url=Config.CELERY_BROKER_URL,
    result_backend=Config.CELERY_RESULT_BACKEND,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(name="process_image", bind=True)
def process_image_task(self, image_path: str):
    try:
        # Call the actual inference function
        result = {"result": image_path}

        return {"status": "success", "task_id": self.request.id, "result": result}
    except Exception as e:
        print(f"[Worker] Error processing image {image_path}: {e}")
        return {"status": "error", "message": str(e)}
