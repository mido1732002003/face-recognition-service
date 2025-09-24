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
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Client │────▶│ FastAPI │────▶│ Face Engine │
└─────────────┘ └─────────────┘ └─────────────┘
│ │
▼ ▼
┌─────────────┐ ┌─────────────┐
│ PostgreSQL │ │ FAISS │
└─────────────┘ └─────────────┘

### Components

- **API Layer**: FastAPI with automatic OpenAPI documentation
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

```bash
# Clone repository
git clone https://github.com/company/face-recognition-service
cd face-recognition-service

# Setup environment
make setup  # Creates .env, installs deps, runs migrations

# Run service
make run