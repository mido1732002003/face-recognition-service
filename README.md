# Face Recognition Service
Production-ready 1:N face recognition (FastAPI • InsightFace/ArcFace • FAISS • PostgreSQL).

## Features
- 1:N identification, multi-face enrollment, quality filtering, optional liveness.
- FAISS Flat/IVF-PQ index; GPU optional (CUDA 11.8); Docker-friendly.
- Health `/health`, metrics `/metrics`, auto docs `/docs`.

## Architecture
Client → FastAPI → Face Engine → (PostgreSQL, FAISS)

## Prerequisites
- Python 3.11+, PostgreSQL 15+, Docker (optional).

## Setup (API)
git clone <repo> && cd face-recognition-service
python -m venv venv && source venv/bin/activate   # Win: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # set DATABASE_URL, SIMILARITY_THRESHOLD, CORS_ORIGINS
python scripts/download_models.py
docker compose up -d postgres
alembic upgrade head && python scripts/init_db.py
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

## Usage (API)
curl -X POST http://127.0.0.1:8000/api/v1/enroll -F "person_id=alice" -F "image=@/path/alice.jpg"
curl -X POST http://127.0.0.1:8000/api/v1/search -F "image=@/path/query.jpg"

## Training Data
dataset/person_x/* images + optional metadata.json (name/id/labels).

## Scripts
- scripts/download_models.py — fetch InsightFace models
- scripts/init_db.py — create tables/seed
- scripts/reset_db.py — drop & recreate schema
- scripts/load_dataset.py — bulk enroll from folders/JSON
- scripts/export_index.py / import_index.py — persist/restore FAISS

## Web UI (port 8081)
UI env: NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000 (or VITE_API_BASE_URL)
uvicorn webui.main:app --reload --host 127.0.0.1 --port 8081
Open http://127.0.0.1:8081; verify requests hit http://127.0.0.1:8000

## Runbook (quick)
DB → `docker compose up -d postgres`; Migrate → `alembic upgrade head`; Init → `python scripts/init_db.py`
API → `uvicorn api.main:app --reload --port 8000`; UI → `uvicorn webui.main:app --reload --port 8081`
Smoke test → open `/docs`, enroll one image, run search.

## Troubleshooting
CORS blocked → set `CORS_ORIGINS=http://127.0.0.1:8081`; bad base URL → UI env must point to `http://127.0.0.1:8000`.
