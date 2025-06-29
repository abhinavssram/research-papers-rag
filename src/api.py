from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os

from main import ask_question, setup_database

app = FastAPI(title="Document RAG API", description="API for querying Documents")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="../ui"), name="static")

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str
    history: list[dict[str, str]] = []

class QuestionResponse(BaseModel):
    question: str
    answer: str

class SetupResponse(BaseModel):
    success: bool
    message: str

@app.get("/")
def read_root():
    """Serve the chat UI at the root path"""
    return FileResponse("../ui/index.html")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/ask", response_model=QuestionResponse)
async def ask_endpoint(request: QuestionRequest):
    """Ask a question about the Documents"""
    try:
        answer = ask_question(request.question, request.history)
        return QuestionResponse(question=request.question, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/setup", response_model=SetupResponse)
async def setup_endpoint():
    """Setup the database with documents and embeddings (run once)"""
    try:
        success = setup_database()
        if success:
            return SetupResponse(success=True, message="Database setup completed successfully!")
        else:
            return SetupResponse(success=False, message="Database setup failed!")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Setup error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
