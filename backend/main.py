from fastapi import FastAPI
from backend.routes import router
app = FastAPI(
    title="AstralVision AI",
    description="AI-powered astronomical image analysis",
    version="1.0.0"
)
app.include_router(router)
@app.get("/")
def root():
    return {
        "message": "AstralVision AI API is running"
    }