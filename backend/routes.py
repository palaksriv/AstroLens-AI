from pathlib import Path
import shutil
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.schemas import AnalysisResponse
from backend.services import analyze_image

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB


@router.get("/health")
def health():
    return {"status": "ok"}


# Plain `def` (not `async def`): the ML pipeline is blocking, so FastAPI runs it
# in a worker thread instead of freezing the event loop.
@router.post("/predict", response_model=AnalysisResponse)
def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a JPG, PNG or WEBP image.",
        )

    # Never trust the client filename (path traversal / overwrites): use a UUID.
    file_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file_path.stat().st_size > MAX_UPLOAD_BYTES:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail="Image is too large (max 15 MB).")

    try:
        return analyze_image(str(file_path))
    except Exception as exc:
        print(f"Analysis failed: {exc!r}")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while analyzing the image. Please try again.",
        )
