# Project Decisions

- **Naming:** product = *AstralVision AI*; repo = `AstroLens-AI`; dataset class = `AstroLensDataset` (single name everywhere in code).
- **Model:** the fine-tuned BLIP (`models/fine_tuned_blip/`) is never replaced by a stock Hugging Face model unless explicitly decided. Checkpoint is git-ignored (trained in Colab).
- **Knowledge base:** NASA APOD (`data/raw/rag/nasa_apod_complete.csv`) → Sentence-BERT `all-MiniLM-L6-v2` → ChromaDB collection `astronomy_knowledge`. Each document stores title/date/APOD URL as metadata so the UI can show a source.
- **LLM:** Gemini (`gemini-flash-latest`), instructed to use only retrieved knowledge and to flag uncertainty.
- **API:** FastAPI, `POST /predict`. Structured response (caption, observation, facts, knowledge cards) so the UI never has to parse raw text or show raw JSON.
- **Frontend:** React + Vite + Tailwind in `frontend/` (replaces the empty Streamlit placeholder that was in `app/`).
- **Secrets:** `GEMINI_API_KEY` lives only in `.env` (see `.env.example`); never printed or committed.
