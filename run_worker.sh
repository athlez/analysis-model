#!/bin/bash
# Script to run the Celery worker properly configured for a GPU server.
# Uses --pool=solo to avoid CUDA multiprocessing fork issues.

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "Starting Celery GPU Worker..."
celery -A app.worker.celery_app worker --pool=solo --loglevel=info
