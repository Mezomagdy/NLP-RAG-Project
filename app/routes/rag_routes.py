"""
Routes (MVC: Controller layer)
Handles HTTP routing. Business logic stays in rag_service.py.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
import shutil
import os

from app.models.schemas import QueryRequest, QueryResponse, UploadResponse, HealthResponse
from app.services.rag_service import run_pipeline

router = APIRouter()

rag_system = None
embedding_client = None
vectordb_client = None

DATA_DIR = "/data"
PROJECT_ID = "project1"

chat_histories = {}


@router.get("/", response_model=HealthResponse)
def health_check():
    """Simple health check to confirm the server is running."""
    index_size = vectordb_client.index.ntotal if vectordb_client else 0
    return HealthResponse(status="ok", index_size=index_size)


@router.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """
    Main RAG endpoint.
    Accepts a question, retrieves relevant chunks, returns LLM answer.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    session_id = request.session_id or "default"
    chat_history = chat_histories.get(session_id, None)

    answer, full_prompt, chat_history = rag_system.answer_rag_question(
        project=PROJECT_ID,
        user_query=request.query,
        top_k=request.top_k,
        chat_history=chat_history
    )

    chat_histories[session_id] = chat_history

    return QueryResponse(
        query=request.query,
        answer=answer,
        full_prompt=full_prompt,
    )


@router.post("/upload", response_model=UploadResponse)
def upload_and_index(files: list[UploadFile] = File(...)):
    """
    Upload new documents and index them into FAISS.
    Accepts PDF, DOCX, HTML, or TXT files.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    saved = 0

    for file in files:
        dest = os.path.join(DATA_DIR, file.filename)
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved += 1

    # Re-run the pipeline on the data folder
    success = run_pipeline(DATA_DIR, PROJECT_ID, embedding_client, vectordb_client)

    if not success:
        raise HTTPException(status_code=422, detail="No valid documents found in upload")

    return UploadResponse(
        message="Files uploaded and indexed successfully",
        files_processed=saved
    )