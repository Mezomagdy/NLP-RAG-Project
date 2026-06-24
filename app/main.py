"""
Main (MVC: App entry point)
Creates the FastAPI app, initializes all clients, registers routes.
"""

from fastapi import FastAPI
import os

from app.services.rag_service import (
    EmbeddingClient,
    FAISSVectorDB,
    LLM_generation,
    Rag_system,
    run_pipeline,
)
from app.routes import rag_routes

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG API",
    description="CV Matching RAG System",
    version="1.0.0"
)

DATA_DIR = "/data"
PROJECT_ID = "project1"


@app.on_event("startup")
def startup_event():
    """
    Runs once when the server starts.
    Initializes embedding model, FAISS DB, LLM, and indexes any existing data.
    """
    print("Initializing RAG system...")

    embedding_client = EmbeddingClient()
    vectordb_client = FAISSVectorDB(dim=embedding_client.embedding_size)
    llm = LLM_generation()
    rag = Rag_system(embedding_client, vectordb_client, llm)

    # Share instances with the routes module
    rag_routes.rag_system = rag
    rag_routes.embedding_client = embedding_client
    rag_routes.vectordb_client = vectordb_client

    # Auto-index documents in /data on startup (if any exist)
    if os.path.isdir(DATA_DIR) and os.listdir(DATA_DIR):
        run_pipeline(DATA_DIR, PROJECT_ID, embedding_client, vectordb_client)
    else:
        print(f"No documents found in {DATA_DIR}. Upload files via POST /upload.")

    print("RAG system ready!")


# ── Register routes ─────────────────────────────────────────────────────────
app.include_router(rag_routes.router)
