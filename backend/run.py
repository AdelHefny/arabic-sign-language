"""
backend/main.py
===============
Uvicorn entry-point for the Arabic Sign Language backend.

Run locally
-----------
    python -m backend.main
    # or
    uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

Run via Docker
--------------
    docker compose up
"""

import uvicorn
from . import config

if __name__ == "__main__":
    uvicorn.run(
        "backend.app:app",
        host=config.HOST,
        port=config.PORT,
        workers=config.WORKERS,
        log_level=config.LOG_LEVEL.lower(),
        reload=False,
    )
