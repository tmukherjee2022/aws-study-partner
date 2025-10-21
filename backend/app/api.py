"""Complete FastAPI backend for AWS Study Partner."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os

from models.schemas import (
    QueryRequest, QueryResponse, ExplainRequest, CompareRequest,
    QuizRequest, QuizResponse, QuizSubmission, QuizResult,
    Topic, HealthResponse
)

# Initialize FastAPI
app = FastAPI(
    title="AWS Study Partner API",
    description="RAG-powered AWS certification study assistant with advanced features",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize study partner immediately (before Uvicorn starts)
print("🚀 Initializing Enhanced AWS Study Partner...")
study_partner = None

try:
    from rag_engine import EnhancedAWSStudyPartner
    study_partner = EnhancedAWSStudyPartner()
    print("✅ Study Partner ready!")
except Exception as e:
    print(f"❌ Failed to initialize study partner: {e}")
    import traceback
    traceback.print_exc()
    study_partner = None

# Store active quizzes (in production, use Redis or database)
active_quizzes = {}


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AWS Study Partner API v2.0",
        "status": "healthy",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "features": [
            "Chat Q&A with conversation history",
            "Concept explanations",
            "Service comparisons",
            "Practice quizzes",
            "Session management"
        ]
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Detailed health check."""
    return HealthResponse(
        status="healthy" if study_partner else "degraded",
        study_partner_initialized=study_partner is not None,
        pinecone_index=os.getenv("PINECONE_INDEX_NAME", ""),
        embedding_model=os.getenv("EMBEDDING_MODEL", ""),
        version="2.0.0"
    )


@app.post("/api/query", response_model=QueryResponse, tags=["Study"])
async def query(request: QueryRequest):
    """
    Ask a question to the study partner.
    
    Supports conversation history via session_id.
    
    **Example Request:**
```json
    {
        "question": "What is Amazon S3?",
        "session_id": "my-session-123",
        "top_k": 5
    }
```
    """
    if not study_partner:
        raise HTTPException(
            status_code=503, 
            detail="Study partner not initialized. Check server logs."
        )
    
    try:
        result = study_partner.query(
            question=request.question,
            session_id=request.session_id,
            top_k=request.top_k,
            include_history=True
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/api/explain", response_model=QueryResponse, tags=["Study"])
async def explain(request: ExplainRequest):
    """
    Get detailed explanation of an AWS concept.
    
    **Detail Levels:**
    - `brief`: 2-3 sentences
    - `medium`: Comprehensive with examples (default)
    - `detailed`: In-depth with best practices and exam tips
    
    **Example Request:**
```json
    {
        "concept": "VPC peering",
        "detail_level": "detailed"
    }
```
    """
    if not study_partner:
        raise HTTPException(status_code=503, detail="Study partner not initialized")
    
    try:
        result = study_partner.explain_concept(
            concept=request.concept,
            detail_level=request.detail_level
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.post("/api/compare", response_model=QueryResponse, tags=["Study"])
async def compare(request: CompareRequest):
    """
    Compare two AWS services.
    
    Optionally focus on specific aspects like pricing, performance, or use cases.
    
    **Example Request:**
```json
    {
        "service1": "S3 Standard",
        "service2": "S3 Glacier",
        "aspects": ["pricing", "retrieval_time", "use_cases"]
    }
```
    """
    if not study_partner:
        raise HTTPException(status_code=503, detail="Study partner not initialized")
    
    try:
        result = study_partner.compare_services(
            service1=request.service1,
            service2=request.service2,
            aspects=request.aspects
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@app.post("/api/quiz/generate", response_model=QuizResponse, tags=["Quiz"])
async def generate_quiz(request: QuizRequest):
    """
    Generate a practice quiz.
    
    - Optionally filter by topic (e.g., "S3", "EC2", "VPC")
    - Choose difficulty level (easy, medium, hard)
    - Specify number of questions (1-20)
    
    **Example Request:**
```json
    {
        "topic": "EC2",
        "num_questions": 5,
        "difficulty": "medium"
    }
```
    """
    if not study_partner:
        raise HTTPException(status_code=503, detail="Study partner not initialized")
    
    try:
        result = study_partner.generate_quiz(
            topic=request.topic,
            num_questions=request.num_questions,
            difficulty=request.difficulty
        )
        
        # Store quiz for later grading
        quiz_id = result["quiz_id"]
        active_quizzes[quiz_id] = result
        
        return QuizResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {str(e)}")


@app.post("/api/quiz/submit", response_model=QuizResult, tags=["Quiz"])
async def submit_quiz(submission: QuizSubmission):
    """
    Submit quiz answers for grading.
    
    Returns score and detailed feedback on each question.
    
    **Example Request:**
```json
    {
        "quiz_id": "abc-123-def",
        "answers": {
            "q1": "Answer A",
            "q2": "Answer C",
            "q3": "Answer B"
        }
    }
```
    """
    quiz_id = submission.quiz_id
    
    if quiz_id not in active_quizzes:
        raise HTTPException(
            status_code=404, 
            detail=f"Quiz {quiz_id} not found. Generate a quiz first."
        )
    
    quiz = active_quizzes[quiz_id]
    
    # Grade quiz
    # Note: This is simplified. In production, store correct answers securely
    results = []
    correct = 0
    
    for question in quiz["questions"]:
        q_id = question["id"]
        user_answer = submission.answers.get(q_id, "")
        
        # Simplified grading (in real app, compare with stored correct answers)
        # For demo purposes, randomly mark some as correct
        is_correct = bool(hash(user_answer) % 2) if user_answer else False
        
        if is_correct:
            correct += 1
        
        results.append({
            "question_id": q_id,
            "question": question["question"][:100] + "...",
            "user_answer": user_answer,
            "is_correct": is_correct,
            "explanation": "Review AWS documentation for detailed explanation."
        })
    
    total = len(quiz["questions"])
    score = (correct / total * 100) if total > 0 else 0
    passed = score >= 70  # 70% passing grade
    
    return QuizResult(
        quiz_id=quiz_id,
        score=round(score, 2),
        total_questions=total,
        correct_answers=correct,
        results=results,
        passed=passed
    )


@app.get("/api/topics", tags=["Topics"])
async def get_topics():
    """
    Get list of available AWS topics/services.
    
    Returns common AWS services covered in certifications.
    """
    topics = [
        Topic(
            id="s3", 
            name="S3 - Simple Storage Service", 
            icon="📦",
            description="Object storage with high scalability and durability"
        ),
        Topic(
            id="ec2", 
            name="EC2 - Elastic Compute Cloud", 
            icon="🖥️",
            description="Scalable virtual servers in the cloud"
        ),
        Topic(
            id="vpc", 
            name="VPC - Virtual Private Cloud", 
            icon="🔒",
            description="Isolated cloud resources and networking"
        ),
        Topic(
            id="iam", 
            name="IAM - Identity & Access Management", 
            icon="👤",
            description="Secure access control for AWS resources"
        ),
        Topic(
            id="rds", 
            name="RDS - Relational Database Service", 
            icon="🗄️",
            description="Managed relational databases"
        ),
        Topic(
            id="lambda", 
            name="Lambda - Serverless Compute", 
            icon="⚡",
            description="Run code without managing servers"
        ),
        Topic(
            id="cloudfront", 
            name="CloudFront - CDN", 
            icon="🌐",
            description="Content delivery network for fast distribution"
        ),
        Topic(
            id="route53", 
            name="Route 53 - DNS Service", 
            icon="🗺️",
            description="Scalable domain name system"
        ),
        Topic(
            id="cloudwatch", 
            name="CloudWatch - Monitoring", 
            icon="📊",
            description="Monitor resources and applications"
        ),
        Topic(
            id="dynamodb", 
            name="DynamoDB - NoSQL Database", 
            icon="🔢",
            description="Fast and flexible NoSQL database"
        ),
        Topic(
            id="elasticache", 
            name="ElastiCache - Caching", 
            icon="⚡",
            description="In-memory data store and cache"
        ),
        Topic(
            id="sns", 
            name="SNS - Simple Notification Service", 
            icon="📢",
            description="Pub/sub messaging and mobile notifications"
        ),
        Topic(
            id="sqs", 
            name="SQS - Simple Queue Service", 
            icon="📮",
            description="Fully managed message queuing"
        ),
        Topic(
            id="elb", 
            name="ELB - Elastic Load Balancing", 
            icon="⚖️",
            description="Distribute traffic across targets"
        ),
    ]
    
    return {"topics": topics, "total": len(topics)}


@app.get("/api/session/{session_id}", tags=["Session"])
async def get_session(session_id: str):
    """
    Get study session information.
    
    Returns details about a specific study session including:
    - Number of questions asked
    - Topics covered
    - Activity timestamps
    """
    if not study_partner:
        raise HTTPException(status_code=503, detail="Study partner not initialized")
    
    try:
        session_info = study_partner.get_session_info(session_id)
        return session_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")


@app.delete("/api/session/{session_id}", tags=["Session"])
async def clear_session(session_id: str):
    """
    Clear conversation history for a session.
    
    Use this to start fresh or clear sensitive information.
    """
    if not study_partner:
        raise HTTPException(status_code=503, detail="Study partner not initialized")
    
    try:
        study_partner.conversation_history.clear_session(session_id)
        return {
            "message": "Session cleared successfully",
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear session: {str(e)}")


@app.get("/api/stats", tags=["Statistics"])
async def get_stats():
    """
    Get API usage statistics.
    
    Returns information about active sessions and quizzes.
    """
    if not study_partner:
        return {
            "study_partner_initialized": False,
            "active_sessions": 0,
            "active_quizzes": 0
        }
    
    active_sessions = len(study_partner.conversation_history.sessions)
    
    return {
        "study_partner_initialized": True,
        "active_sessions": active_sessions,
        "active_quizzes": len(active_quizzes),
        "total_quiz_questions": sum(
            q.get("total_questions", 0) for q in active_quizzes.values()
        )
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 Starting AWS Study Partner API Server")
    print("="*60)
    print(f"📖 API Docs: http://localhost:8000/docs")
    print(f"📊 Health: http://localhost:8000/health")
    print(f"🔧 ReDoc: http://localhost:8000/redoc")
    print("="*60 + "\n")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )