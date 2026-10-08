# AstroLens AI

**AI-powered astronomical image analysis** — *Understand the Universe, One Image at a Time.*

AstroLens AI combines computer vision, image captioning, semantic retrieval, and generative AI to analyze astronomical images.

Upload an astronomical image → identify the likely celestial object → generate an image caption → retrieve relevant astronomy knowledge → generate a grounded scientific observation.

---

## Pipeline

```text
                         ┌─────────────────────┐
                         │   Astronomical      │
                         │       Image         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       CLIP          │
                         │ Object Recognition  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Fine-tuned BLIP     │
                         │ Image Captioning    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Sentence-BERT     │
                         │ all-MiniLM-L6-v2    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      ChromaDB       │
                         │ Astronomy Knowledge │
                         │     NASA / APOD     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       Gemini        │
                         │ Scientific Analysis │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Analysis Report   │
                         └─────────────────────┘
How it works
CLIP performs zero-shot astronomical object recognition.
Fine-tuned BLIP generates a natural-language caption of the image.
Sentence-BERT converts the recognized object into a semantic embedding.
ChromaDB retrieves relevant astronomy knowledge from the local knowledge base.
Gemini uses the recognition, caption, and retrieved knowledge to generate a scientific observation and interesting facts.
The React frontend presents the results as an astronomy observation report.
Features
🔭 Astronomical Object Recognition

AstroLens uses OpenAI CLIP for zero-shot recognition against a curated astronomy vocabulary.

Example:

Thor's Helmet Nebula       96.77%
Great Nebula in Cepheus     1.46%
Carina Nebula               1.13%
Trifid Nebula                0.21%
Crab Nebula                  0.14%

The highest-ranked recognition is used as the query for the astronomy knowledge retrieval stage.

🖼️ Fine-tuned Image Captioning

A fine-tuned BLIP model generates a caption describing the uploaded astronomical image.

The trained checkpoint is stored locally in:

models/fine_tuned_blip/

The checkpoint is not included in Git because of its size.

📚 Astronomy Knowledge Retrieval

AstroLens uses:

Sentence-BERT
all-MiniLM-L6-v2
ChromaDB

to retrieve semantically relevant astronomy information.

The knowledge base contains astronomy observations and NASA Astronomy Picture of the Day (APOD) information.

✨ Scientific Analysis

Gemini receives the recognized object, generated caption, and retrieved astronomy knowledge to produce:

Scientific observations
Interesting facts
Contextual explanations

The prompt instructs Gemini to ground the explanation in the retrieved astronomy information.

🌌 Astronomy-focused Interface

The frontend is built as a dark observatory-style interface rather than a generic AI dashboard.

It includes:

Drag-and-drop image upload
Image preview
Object identification
Recognition confidence
Alternative predictions
AI caption
Scientific observation
Interesting facts
Related NASA/APOD observations
Source links
Analyze-another-image workflow
Technology Stack
Frontend
React
Vite
Tailwind CSS
Lucide React
Backend
Python
FastAPI
Uvicorn
Pydantic
AI / Computer Vision
PyTorch
Hugging Face Transformers
OpenAI CLIP
BLIP
Sentence Transformers
Google Gemini API
Retrieval
ChromaDB
Sentence-BERT
NASA/APOD astronomy knowledge
Repository Layout
AstroLens-AI/
│
├── backend/
│   ├── main.py
│   ├── routes.py
│   ├── services.py
│   └── schemas.py
│
├── models/
│   ├── captioning/
│   │   ├── dataset.py
│   │   ├── eval.py
│   │   ├── inference.py
│   │   └── train.py
│   │
│   ├── recognition/
│   │   └── clip.py
│   │
│   ├── rag/
│   │   ├── embeddings.py
│   │   ├── retriever.py
│   │   └── gemini.py
│   │
│   ├── fine_tuned_blip/
│   │   └── model files
│   │
│   └── pipeline.py
│
├── pipelines/
│   ├── preprocess_blip.py
│   ├── validate_blip.py
│   └── inspect_blip_dataset.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── blip/
│   │   └── chroma/
│   └── uploads/
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── sample_images/
│
├── docs/
│   └── dataset_research.md
│
├── .env.example
├── .gitignore
├── PROJECT_DECISIONS.md
├── requirements.txt
└── README.md
Setup
1. Clone the repository
git clone https://github.com/palaksriv/AstroLens-AI.git
cd AstroLens-AI
2. Create the Python environment
python -m venv .venv
Windows
.\.venv\Scripts\activate
Install dependencies
pip install -r requirements.txt
3. Configure Gemini

Create a local .env file:

Copy-Item .env.example .env

Add your Gemini API key:

GEMINI_API_KEY=your_api_key_here

.env is ignored by Git and should never be committed.

Model Setup
1. BLIP Checkpoint

The fine-tuned BLIP checkpoint is not included in Git because of its size.

The expected directory is:

models/fine_tuned_blip/

The directory should contain files such as:

config.json
generation_config.json
model.safetensors
processor_config.json
tokenizer.json
tokenizer_config.json

The model can be restored from the trained checkpoint or retrained.

Retraining

If regenerating the processed dataset:

python pipelines/preprocess_blip.py

Validate the dataset:

python pipelines/validate_blip.py

Train BLIP:

python models/captioning/train.py

Evaluate the model:

python models/captioning/eval.py

The training pipeline saves the best checkpoint to:

models/fine_tuned_blip/
2. ChromaDB Knowledge Base

The processed ChromaDB database is not included in Git.

Build it with:

python -m models.rag.embeddings

This creates the local astronomy knowledge collection:

astronomy_knowledge
Running AstroLens

AstroLens uses two local development servers.

Terminal 1 — Backend

From the project root:

uvicorn backend.main:app --reload

Backend:

http://127.0.0.1:8000

Health check:

GET /health
Terminal 2 — Frontend
cd frontend
npm install
npm run dev

Frontend:

http://localhost:5173

Open the frontend in your browser and upload an astronomical image.

API
Health
GET /health

Response:

{
  "status": "ok"
}
Analyze Image
POST /predict

The endpoint accepts a multipart form field named:

file

Supported formats:

JPG
JPEG
PNG
WEBP

Maximum file size:

15 MB

Example:

curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@sample_images/images (1).jpeg"
Response

The API returns:

{
  "recognition": [
    {
      "label": "Thor's Helmet Nebula",
      "confidence": 0.9677
    }
  ],
  "caption": "the great nebula in cepheus",
  "observation": "...",
  "scientific_observation": "...",
  "facts": [
    "..."
  ],
  "documents": [
    "..."
  ],
  "knowledge": [
    {
      "title": "Thor's Helmet",
      "summary": "...",
      "date": "2010-06-05",
      "source_url": "https://apod.nasa.gov/apod/ap100605.html"
    }
  ]
}
Example Pipeline Output

For a Thor's Helmet Nebula image, the current recognition system produces:

Primary identification:
Thor's Helmet Nebula

Confidence:
96.77%

Alternative predictions:
Great Nebula in Cepheus
Carina Nebula
Trifid Nebula
Crab Nebula

The retrieval system then searches the astronomy knowledge base using:

Thor's Helmet Nebula

and retrieves relevant NASA/APOD observations.

BLIP Training

The BLIP model was fine-tuned using the astronomy image-caption dataset.

The best validation checkpoint was obtained at Epoch 4.

Epoch 1
Train Loss: 3.9240
Validation Loss: 3.3510

Epoch 2
Train Loss: 2.7571
Validation Loss: 3.0955

Epoch 3
Train Loss: 1.9574
Validation Loss: 2.9672

Epoch 4
Train Loss: 1.3334
Validation Loss: 2.9197

Epoch 5
Train Loss: 0.8715
Validation Loss: 2.9385

The Epoch 4 checkpoint achieved the lowest validation loss and was retained for inference.

Current Status
Working
 React frontend
 Vite development environment
 Tailwind CSS interface
 Astronomy-focused UI
 Image upload
 Image preview
 FastAPI backend
 Image validation
 CLIP zero-shot recognition
 Recognition confidence scores
 Fine-tuned BLIP inference
 Sentence-BERT embeddings
 ChromaDB retrieval
 NASA/APOD knowledge retrieval
 Gemini integration
 API response pipeline
 Frontend → FastAPI integration
External Dependency

Gemini is an external API service. During periods of high demand, the Gemini API may temporarily return 503 Service Unavailable.

AstroLens handles this situation with a fallback response so that the recognition, captioning, and knowledge retrieval stages remain available even when Gemini is temporarily unavailable.

Design Goals

AstroLens is built around three core ideas:

Visual Understanding

Identify what is present in an astronomical image using computer vision.

Grounded Knowledge

Connect visual recognition to a searchable astronomy knowledge base.

Accessible Astronomy

Present astronomical information in a way that is understandable to students, astronomy enthusiasts, and general users.

Future Improvements
Expanded astronomical object vocabulary
Improved astronomy-specific CLIP recognition
Additional astronomy datasets
Improved BLIP caption quality
More NASA datasets
Object-specific scientific analysis
Image similarity search
Observation history
Improved Gemini retry handling
GPU inference optimization
Cloud deployment