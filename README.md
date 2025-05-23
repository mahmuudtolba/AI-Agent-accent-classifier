# AI Agent Accent Classifier

This project provides an accent classification API service powered by a deep learning model for speech. It uses a pretrained speech model to recognize and classify the accent of English speakers from audio input. The service is built with FastAPI for high-performance REST endpoints and is containerized with Docker. It can classify an input audio clip and return the predicted accent along with confidence scores, making it useful for applications like voice profiling, call routing, or multilingual customer support.

## Features

* **Real-time Accent Classification:** Detects and classifies the accent in an input audio clip using a pretrained speech model.
* **REST API Endpoints:** Provides a health-check endpoint (`GET /`) and a classification endpoint (e.g. `POST /api/v1/accent_classification`) for processing requests.
* **JSON Input/Output:** Accepts JSON payloads containing audio URLs and returns JSON with predicted accents, confidence scores, and processing latency.
* **Dockerized Deployment:** Supports containerized deployment using Docker and Docker Compose. Includes an Nginx container as a reverse proxy listening on port 80.
* **Pretrained Model Management:** Loads pretrained accent classification model weights (e.g. from Hugging Face or cloud storage) at runtime for inference.
* **Easy Extensibility:** The backend code is modular (FastAPI routes, Pydantic models) making it straightforward to extend or integrate additional functionality.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/mahmuudtolba/AI-Agent-accent-classifier.git
   cd AI-Agent-accent-classifier
   ```
2. **Install Python dependencies (local):**
   Ensure you have Python 3.11+ installed. (Optionally create and activate a virtual environment.) Then run:

   ```bash
   pip install -r requirements.txt
   ```
3. **Run the service locally:**
   Use Uvicorn to start the FastAPI app:

   ```bash
   chmod +x /app/entrypoint.sh
   /entrypoint.sh
   ```
4. **Run with Docker Compose (recommended):**
   Ensure Docker and Docker Compose are installed. Then build and start the containers:

   ```bash
   docker-compose up --build
   ```

   This will start the API and an Nginx reverse proxy.

## Usage

* **Health Check:** Send a `GET` request to `/` to verify the service is running (e.g. returns status or a simple message).
* **Accent Classification API:** Send a `POST` request to `/api/v1/accent_classification` with a JSON body. For example:

  ```json
  {
    "video_url": ["https://example.com/path/to/audio.wav"]
  }
  ```

  The API will process the audio and respond with a JSON object containing the predicted accent(s) and confidence score(s). Example response (conceptual):

  ```json
  {
    "predictions": [
      {"accent": "American", "confidence": 0.92}
    ],
    "latency": 0.15
  }
  ```

## Project Structure

* `backend/` – FastAPI application code (API endpoints, model loading, inference logic).
* `frontend/` – (Optional) Frontend UI code or demo interfaces (e.g. notebooks or web interface for testing).
* `development_notebooks/` – Jupyter notebooks for data exploration, model training, and evaluation used during development.
* `Dockerfile`, `Dockerfile.nginx`, `nginx.conf` – Container and deployment configuration files for the application and reverse proxy.
* `docker-compose.yml` – Docker Compose file to orchestrate the API service and Nginx proxy in multi-container setup.
* `.dockerignore`, `.gitignore` – Files specifying patterns to ignore in Docker builds and Git.
* `requirements.txt` – Python dependencies required to run the project (FastAPI, PyTorch, audio processing libraries, etc.).

## Technologies Used

* **FastAPI** – Web framework for building the REST API.
* **Uvicorn** – ASGI server for running the FastAPI app.
* **Python 3.11** – Programming language.
* **PyTorch** (or TensorFlow) – Deep learning framework for loading the pretrained accent model.
* **Hugging Face Transformers** – (If used) Pretrained speech models (e.g. Wav2Vec2) for audio feature extraction and classification.
* **Pydantic** – Data validation for request and response models.
* **Librosa / Torchaudio** – Libraries for audio processing (e.g. loading and feature extraction from audio).
* **Docker & Docker Compose** – Containerization and orchestration tools for deployment.
* **Nginx** – Reverse proxy server to route HTTP requests (configured for Docker deployment).
* **Additional Libraries:** Requests (HTTP client), NumPy/Pandas (data handling), etc.
