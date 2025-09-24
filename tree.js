// tree.js
const fs = require("fs");
const path = require("path");

const files = [
  ".editorconfig",
  ".env.example",
  ".gitignore",
  ".pre-commit-config.yaml",
  "LICENSE",
  "Makefile",
  "README.md",
  "Dockerfile",
  "docker-compose.yml",
  "pyproject.toml",
  "requirements.txt",
  "ruff.toml",
  "alembic.ini",
  "alembic/env.py",
  "alembic/script.py.mako",
  "alembic/versions/001_initial_schema.py",
  "api/__init__.py",
  "api/main.py",
  "api/config.py",
  "api/dependencies.py",
  "api/middleware.py",
  "api/routers/__init__.py",
  "api/routers/enrollment.py",
  "api/routers/identification.py",
  "api/routers/persons.py",
  "api/routers/health.py",
  "api/routers/metrics.py",
  "core/__init__.py",
  "core/models.py",
  "core/schemas.py",
  "core/database.py",
  "core/exceptions.py",
  "core/constants.py",
  "services/__init__.py",
  "services/face_engine.py",
  "services/face_quality.py",
  "services/liveness.py",
  "services/enrollment_service.py",
  "services/identification_service.py",
  "services/person_service.py",
  "services/background_jobs.py",
  "indexing/__init__.py",
  "indexing/base.py",
  "indexing/faiss_index.py",
  "indexing/ivfpq_index.py",
  "indexing/scann_adapter.py",
  "indexing/milvus_adapter.py",
  "indexing/qdrant_adapter.py",
  "utils/__init__.py",
  "utils/logging.py",
  "utils/metrics.py",
  "utils/image_utils.py",
  "scripts/download_models.py",
  "scripts/init_db.py",
  "scripts/migrate.py",
  "scripts/seed_data.py",
  "scripts/reindex.py",
  "storage/index/.gitkeep",
  "data/.gitkeep",
  "logs/.gitkeep",
  "tests/__init__.py",
  "tests/conftest.py",
  "tests/fixtures/mock_faces.py",
  "tests/unit/test_face_engine.py",
  "tests/unit/test_indexing.py",
  "tests/integration/test_enrollment.py",
  "tests/integration/test_identification.py",
  "tests/integration/test_health.py",
  "notebooks/threshold_calibration.ipynb",
  "notebooks/batch_enrollment.ipynb",
  "notebooks/kaggle_deployment.ipynb",
  "docs/architecture.md",
  "docs/optimization.md",
  "docs/postman_collection.json",
  "clients/python/example.py",
  "clients/nodejs/example.js",
  "clients/nodejs/package.json",
  "deployments/kubernetes/deployment.yaml",
  "deployments/helm/face-recognition/Chart.yaml",
  "deployments/helm/face-recognition/values.yaml",
  "monitoring/prometheus.yml",
];

function ensureDir(filePath) {
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

files.forEach((file) => {
  const filePath = path.join(__dirname, "face-recognition-service", file);
  ensureDir(filePath);
  if (!fs.existsSync(filePath)) {
    fs.writeFileSync(filePath, "");
    console.log("Created:", filePath);
  }
});
