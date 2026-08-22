# Face Recognition Starter (Learning Project)

A beginner-friendly face recognition pipeline built to learn **embeddings**,
**vector databases**, and **computer vision** fundamentals — end to end, from
raw photos to nearest-neighbor face matching.

This project detects faces in photos, converts each face into a numeric
"fingerprint" (embedding) using CLIP, stores those fingerprints in a
PostgreSQL database with the `pgvector` extension, and then finds the closest
matching stored face for a new, unseen photo.

## What This Project Covers

- **Computer vision**: face detection using OpenCV's Haar Cascade classifier
- **Embeddings**: converting cropped face images into 768-dimensional vectors
  using CLIP (`openai/clip-vit-base-patch32` via Hugging Face `transformers`)
- **Vector databases**: storing and searching those vectors with PostgreSQL +
  `pgvector`, running locally in Docker
- **Nearest-neighbor search**: matching a new face against stored faces using
  vector distance operators

## Pipeline Overview

1. **Detect & crop faces** — `cv2.CascadeClassifier` scans a source photo,
   finds face bounding boxes, and saves each cropped face to
   `public/stored-faces/`.
2. **Generate & store embeddings** — Each cropped face is passed through CLIP
   (`model.vision_model(...).pooler_output`, 768 dimensions) and the
   resulting vector is inserted into the `pictures` table in Postgres.
3. **Match a new face** — A new, previously-unseen photo goes through the same
   detect → crop → embed steps, then a `pgvector` distance query finds the
   closest stored embedding and returns the matching filename.

## Prerequisites

- Python 3.12+ with a virtual environment
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (with
  WSL2 backend on Windows)
- pgAdmin 4 (bundled with most PostgreSQL installers, used here purely as a
  GUI for the Dockerized database)

## Database Setup (Docker + pgvector)

This project runs PostgreSQL with the `pgvector` extension pre-installed via
the official Docker image, avoiding the need to compile `pgvector` manually.

```bash
docker run -d --name pgvector-db -e POSTGRES_PASSWORD=yourpassword -p 5433:5432 pgvector/pgvector:pg17
```

Notes:
- The container exposes Postgres on **port 5433** on the host machine, to
  avoid conflicting with a native Postgres install already using the default
  port 5432.
- Verify the container is running: `docker ps`
- Restart it later (after a reboot) with: `docker start pgvector-db`

Connect to it with pgAdmin 4 (Host: `localhost`, Port: `5433`, User:
`postgres`), then create the database and enable the extension:

```sql
CREATE EXTENSION vector;

CREATE TABLE pictures (
    picture TEXT,
    embedding VECTOR(768)
);
```

## Environment Variables

Create a `.env` file in the project root (excluded from version control):

```
POSTGRES_PASSWORD=yourpassword
```

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
opencv-python<5
transformers
torch
psycopg2-binary
pgvector
pillow
python-dotenv
```

Also download the Haar Cascade face-detection model file into the project
root:

```bash
curl -o haarcascade_frontalface_default.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
```

## Usage

Run the notebook (`face_recog.ipynb`) top to bottom, or run the equivalent
steps as scripts:

1. **Crop faces** from a source photo into `public/stored-faces/`.
2. **Embed and store** each cropped face into the `pictures` table.
3. **Query a new photo** — detect its face, generate its embedding, and run
   a nearest-neighbor search:

```python
cur.execute(
    "SELECT picture FROM pictures ORDER BY embedding <=> %s LIMIT 1",
    (str(embedding.tolist()),)
)
```

`<=>` is pgvector's cosine distance operator — it compares the *direction* of
two embeddings rather than their raw magnitude, which fits CLIP embeddings
better than Euclidean distance (`<->`).

## Key Concepts (Quick Reference)

- **Embedding** — a list of numbers representing an object (here, a face) such
  that similar objects produce numerically similar lists.
- **Vector** — the general math term for that list of numbers; "embedding"
  and "vector" describe the same object.
- **CLIP** — a pretrained neural network (from OpenAI) used here purely for
  inference (no training performed) to convert face images into 768-number
  vectors.
- **pgvector** — a PostgreSQL extension adding a native `VECTOR` column type
  and distance operators (`<->` Euclidean, `<=>` cosine) for similarity
  search directly in SQL.

## Known Gotchas

- `imgbeddings` (an earlier approach used in this project) breaks on newer
  `huggingface_hub` versions due to a removed `cached_download` function.
  This project uses `transformers`' `CLIPModel`/`CLIPProcessor` directly
  instead, which avoids that dependency conflict.
- Each new script/session needs its own fresh `psycopg2.connect(...)` call —
  reusing a `conn` object closed in an earlier cell/script raises
  `InterfaceError: connection already closed`.
- `model.vision_model(...).pooler_output` returns 768 dimensions, matching
  this project's `VECTOR(768)` column. `model.get_image_features()` returns a
  different, 512-dimensional projected vector instead — the two are not
  interchangeable without changing the table schema.

## Project Status

Currently a learning sandbox, not a production system. Next possible steps:
webcam-based live recognition, swapping `pgvector`'s exact search for an
indexed approximate search (IVFFlat/HNSW) at larger scale, and building a
small gallery-management interface.
