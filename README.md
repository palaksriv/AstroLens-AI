# AstralVision AI

**AI-powered astronomy image analysis** — *Understand the Universe, One Image at a Time.*

Upload an astronomical image → the AI captions it → relevant NASA/APOD knowledge is retrieved → Gemini writes a grounded scientific explanation.

> Naming: **AstralVision AI** is the product name. The GitHub repo is `AstroLens-AI` and internal code identifiers (e.g. `AstroLensDataset`) use the AstroLens name.

## Pipeline

```
Image → fine-tuned BLIP → caption → Sentence-BERT (all-MiniLM-L6-v2)
      → ChromaDB (astronomy_knowledge, NASA APOD) → Gemini → observation + facts
```

## Repository layout

```
backend/     FastAPI app (main.py, routes.py, services.py, schemas.py)
models/      captioning/ (BLIP train/eval/inference), rag/ (embeddings, retriever, gemini), pipeline.py
pipelines/   BLIP data preprocessing / validation / inspection
data/        raw + processed datasets, split_dataset.py
frontend/    React + Vite + Tailwind UI (next phase)
```

## Setup (from a fresh clone)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
copy .env.example .env            # then put your GEMINI_API_KEY in .env
```

### 1. BLIP checkpoint (not in git)
`models/fine_tuned_blip/` is git-ignored because it was trained in Colab. Retrain or restore it, then extract so that `models/fine_tuned_blip/config.json` exists.

To (re)train:
```bash
python pipelines/preprocess_blip.py      # only if regenerating: raw images -> data/processed/blip/images
python pipelines/validate_blip.py
python models/captioning/train.py        # saves best checkpoint to models/fine_tuned_blip/
python models/captioning/eval.py         # optional: pretrained vs fine-tuned metrics
```

### 2. Knowledge base (not in git)
`data/processed/chroma/` is also git-ignored — build it once:
```bash
python -m models.rag.embeddings
```

### 3. Run the API
```bash
uvicorn backend.main:app --reload
```
- `GET /` and `GET /health` – status
- `POST /predict` – multipart field `file` (JPG/PNG/WEBP, ≤15 MB)

Response:
```json
{
  "caption": "...",
  "observation": "full Gemini markdown",
  "scientific_observation": "...",
  "facts": ["...", "...", "..."],
  "documents": ["raw retrieved text"],
  "knowledge": [{"title": "...", "summary": "...", "date": "1995-06-16", "source_url": "https://apod.nasa.gov/apod/ap950616.html"}]
}
```
