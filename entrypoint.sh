#!/bin/bash

# Run FastAPI on port 8000
uvicorn backend.app:app --host 0.0.0.0 --port 8000 &

# Run Streamlit on port 8501
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 &

# Wait for all background jobs to finish
wait