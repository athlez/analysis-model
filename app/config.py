import os


class Config:
    # Replace the default 'localhost' with your remote Redis IP or domain.
    # Format: redis://[:password]@hostname:port/db_number
    # Example: redis://:mypassword@192.168.1.50:6379/0
    
    CELERY_BROKER_URL = os.getenv(
        "CELERY_BROKER_URL", "redis://localhost:6379/0"
    )
    CELERY_RESULT_BACKEND = os.getenv(
        "CELERY_RESULT_BACKEND", "redis://localhost:6379/0"
    )
