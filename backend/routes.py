from fastapi import APIRouter, UploadFile, File
from pathlib import Path
import shutil
from backend.services import analyze_image
router = APIRouter()
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = analyze_image(str(file_path))
    return result