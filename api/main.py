"""
SmartFlow FastAPI Main Application.
Serves the REST API for Person 4's React + Plotly frontend.
Run with:
    uvicorn api.main:app --reload
Base URL:
    http://127.0.0.1:8000
Docs:
    http://127.0.0.1:8000/docs
Dashboard snapshot:
    http://127.0.0.1:8000/api/v1/dashboard/snapshot
"""

import sys
from pathlib import Path

# Add project root to sys.path so project modules resolve reliably
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.dashboard import router as dashboard_router

app = FastAPI(
    title="SmartFlow Execution & SOR API",
    description="FastAPI Backend adapter for the SmartFlow Quantitative Trading & Execution Dashboard.",
    version="1.0.0",
)

# Configure CORS for local Vite development
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(dashboard_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
