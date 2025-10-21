"""Pydantic models for request/response validation."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum


class StudyMode(str, Enum):
    """Available study modes."""
    CHAT = "chat"
    QUIZ = "quiz"
    EXPLAIN = "explain"
    COMPARE = "compare"
    FLASHCARD = "flashcard"


class QueryRequest(BaseModel):
    """Request model for general queries."""
    question: str = Field(..., min_length=3, max_length=1000)
    session_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=10)
    include_sources: bool = True


class Source(BaseModel):
    """Source document information."""
    text: str
    source: str
    doc_type: str
    chunk_id: int
    relevance_score: Optional[float] = None


class QueryResponse(BaseModel):
    """Response model for queries."""
    question: str
    answer: str
    sources: List[Source]
    num_sources: int
    session_id: Optional[str] = None
    processing_time_ms: Optional[float] = None


class ExplainRequest(BaseModel):
    """Request for concept explanation."""
    concept: str = Field(..., min_length=2, max_length=200)
    detail_level: str = Field(default="medium", pattern="^(brief|medium|detailed)$")


class CompareRequest(BaseModel):
    """Request for service comparison."""
    service1: str = Field(..., min_length=2, max_length=100)
    service2: str = Field(..., min_length=2, max_length=100)
    aspects: Optional[List[str]] = None  # e.g., ["pricing", "performance", "use_cases"]


class QuizQuestion(BaseModel):
    """Individual quiz question."""
    id: str
    question: str
    options: Optional[List[str]] = None
    correct_answer: Optional[str] = None
    explanation: Optional[str] = None
    difficulty: Optional[str] = None
    topic: Optional[str] = None


class QuizRequest(BaseModel):
    """Request for quiz generation."""
    topic: Optional[str] = None
    num_questions: int = Field(default=5, ge=1, le=20)
    difficulty: Optional[str] = Field(default=None, pattern="^(easy|medium|hard)?$")


class QuizResponse(BaseModel):
    """Response with quiz questions."""
    quiz_id: str
    topic: str
    questions: List[QuizQuestion]
    total_questions: int


class QuizSubmission(BaseModel):
    """User's quiz answers."""
    quiz_id: str
    answers: Dict[str, str]  # question_id: user_answer


class QuizResult(BaseModel):
    """Quiz grading results."""
    quiz_id: str
    score: float
    total_questions: int
    correct_answers: int
    results: List[Dict]
    passed: bool


class Topic(BaseModel):
    """AWS topic/service."""
    id: str
    name: str
    icon: str
    description: Optional[str] = None


class StudySession(BaseModel):
    """Study session information."""
    session_id: str
    created_at: str
    last_active: str
    questions_asked: int
    topics_covered: List[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    study_partner_initialized: bool
    pinecone_index: str
    embedding_model: str
    version: str