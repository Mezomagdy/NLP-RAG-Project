"""
Models (MVC: Model layer)
Pydantic schemas for request and response validation.
"""

from pydantic import BaseModel
from typing import Optional, List


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    session_id: str = "default"


class QueryResponse(BaseModel):
    query: str
    answer: str
    full_prompt: str


class UploadResponse(BaseModel):
    message: str
    files_processed: int


class HealthResponse(BaseModel):
    status: str
    index_size: int