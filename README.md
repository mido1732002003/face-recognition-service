# Face Recognition Service

Enterprise-grade 1:N face recognition service with production-ready features including face detection, quality assessment, liveness detection, and scalable vector search.

## Features

- **1:N Face Identification**: Search faces across large galleries  
- **Face Enrollment**: Register multiple faces per person  
- **Quality Assessment**: Automatic face quality scoring  
- **Liveness Detection**: Anti-spoofing capabilities (pluggable)  
- **Scalable Index**: FAISS-based vector search with IVF-PQ support  
- **Production Ready**: Async API, metrics, health checks, logging  
- **GPU Support**: Optional CUDA acceleration  
- **Containerized**: Docker and Kubernetes ready  

## Architecture

┌─────────────┐     ┌─────────────┐     ┌─────────────┐  
│   Client    │ ───▶│   FastAPI   │ ───▶│ Face Engine │  
└─────────────┘     └─────────────┘     └─────────────┘  
       │                   │  
       ▼                   ▼  
┌─────────────┐     ┌─────────────┐  
│ PostgreSQL  │     │    FAISS    │  
└─────────────┘     └─────────────┘  

### Components

- **API Layer**: FastAPI with automatic OpenAPI docs  
- **Face Engine**: InsightFace with ArcFace embeddings (L2-normalized)  
- **Detection**: SCRFD (default) or RetinaFace  
- **Vector Index**: FAISS (Flat/IVF-PQ) with adapters for ScaNN/Milvus  
- **Database**: PostgreSQL for metadata storage  
- **Monitoring**: Prometheus metrics + Grafana dashboards  

## Quick Start

### Prerequisites

- Python 3.11+  
- PostgreSQL 15+  
- Docker & Docker Compose (optional)  
- NVIDIA GPU + CUDA 11.8 (optional)  

### Local Installation

    # Clone repository
    git clone https://github.com/company/face-recognition-service
    cd face-recognition-service

    # Setup environment
    make setup  # Creates .env, installs deps, runs migrations

    # Run service
    make run

### Docker Compose

    docker compose up -d

Service will be available at: http://localhost:8000  
API docs: http://localhost:8000/docs  

## Usage

### Enroll a Face

    curl -X POST http://localhost:8000/api/v1/enroll \
      -F "person_id=123" \
      -F "image=@/path/to/image.jpg"

### Search a Face

    curl -X POST http://localhost:8000/api/v1/search \
      -F "image=@/path/to/query.jpg"

### Health Check

    curl http://localhost:8000/health

## Training Data

To prepare training/enrollment data:

    dataset/
    ├── person_1/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── metadata.json
    ├── person_2/
    │   ├── img1.jpg
    │   └── metadata.json
    └── person_3/
        └── ...

- Each **person** has a folder with images.  
- `metadata.json` contains info like name, id, labels. Example:

    {
      "id": "person_1",
      "name": "Alice Johnson",
      "department": "Engineering"
    }

## Deployment

### Docker Image

    docker build -t face-recognition-service .
    docker run -p 8000:8000 face-recognition-service

### Kubernetes

- Includes manifests in `deploy/k8s/` for production.  
- Supports Horizontal Pod Autoscaler.  
- Configurable storage backends (PVC for PostgreSQL, FAISS persistence).  

## Development

    # Lint code
    make lint

    # Run tests
    make test

    # Format code
    make fmt

## Monitoring

- **Prometheus** metrics exposed at `/metrics`.  
- **Grafana** dashboards included in `monitoring/`.  

## Self-test Checklist

- [ ] Run `make setup` and confirm `.env` is created  
- [ ] Start service with `make run` or Docker Compose  
- [ ] Visit http://localhost:8000/docs and test endpoints  
- [ ] Enroll a sample image and confirm it appears in PostgreSQL + FAISS  
- [ ] Perform a search and verify similarity results  

## Scripts

- **scripts/download_models.py** → Downloads required InsightFace models locally  
- **scripts/init_db.py** → Initializes PostgreSQL database with required tables  
- **scripts/reset_db.py** → Drops and recreates database schema (useful for clean state)  
- **scripts/load_dataset.py** → Bulk enrollment of a dataset (folders & JSON metadata)  
- **scripts/benchmark.py** → Runs performance benchmarks on search/index  
- **scripts/export_index.py** → Saves FAISS index to disk for persistence  
- **scripts/import_index.py** → Loads FAISS index from saved file  

## Runbook (Simplified)

### Prerequisites
- Python 3.11+ and pip  
- Docker & Docker Compose  
- Git  
- (Optional) NVIDIA drivers + CUDA 11.8+ for GPU  

### Installation & Setup
    git clone <repository>
    cd face-recognition-service
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    python scripts/download_models.py

### Database setup
    docker compose up -d postgres
    alembic upgrade head
    python scripts/init_db.py

### Environment
    cp .env.example .env
    # Edit DATABASE_URL and other variables

### Run the service
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Access:
- API docs → http://localhost:8000/docs  
- Health → http://localhost:8000/health  
- Metrics → http://localhost:8000/metrics  

### Quick test
    curl -X POST "http://localhost:8000/api/v1/enroll/test_user" -F "images=@test.jpg"
    curl -X POST "http://localhost:8000/api/v1/search" -F "image=@query.jpg"

## License

MIT License. See LICENSE for details.
