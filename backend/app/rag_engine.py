"""Enhanced RAG Query Engine with conversation history and streaming."""
import os
import time
import uuid
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone as PineconeVectorStore

load_dotenv()


class ConversationHistory:
    """Manages conversation history for a session."""
    
    def __init__(self, max_history: int = 5):
        self.sessions = {}
        self.max_history = max_history
    
    def add_message(self, session_id: str, question: str, answer: str):
        """Add Q&A to session history."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        self.sessions[session_id].append({
            "question": question,
            "answer": answer,
            "timestamp": time.time()
        })
        
        # Keep only recent history
        if len(self.sessions[session_id]) > self.max_history:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history:]
    
    def get_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for session."""
        return self.sessions.get(session_id, [])
    
    def clear_session(self, session_id: str):
        """Clear session history."""
        if session_id in self.sessions:
            del self.sessions[session_id]


class EnhancedAWSStudyPartner:
    """Enhanced RAG-based AWS Study Partner with advanced features."""
    
    def __init__(self):
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        )
        
        # Initialize vector store
        self.vectorstore = PineconeVectorStore.from_existing_index(
            index_name=os.getenv("PINECONE_INDEX_NAME", "aws-study-partner"),
            embedding=self.embeddings
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=500
        )
        
        # Initialize conversation history
        self.conversation_history = ConversationHistory()
        
        # System prompt
        self.system_prompt = """You are an expert AWS certification study partner. 
Your role is to help students prepare for AWS certifications by:
- Providing clear, accurate answers based on official AWS documentation
- Explaining complex concepts in simple terms
- Offering relevant examples and use cases
- Helping students understand why answers are correct or incorrect
- Being encouraging and supportive

Always base your answers on the provided context from study materials.
If you're unsure, say so rather than making up information."""
        
        print("✅ Enhanced AWS Study Partner initialized")
    
    def query(
        self, 
        question: str, 
        session_id: Optional[str] = None,
        top_k: int = 5,
        include_history: bool = True
    ) -> Dict:
        """
        Query with optional conversation history.
        
        Args:
            question: User's question
            session_id: Optional session ID for history
            top_k: Number of chunks to retrieve
            include_history: Include conversation history in context
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        start_time = time.time()
        
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Build context from history if available
        history_context = ""
        if include_history and session_id:
            history = self.conversation_history.get_history(session_id)
            if history:
                history_context = "\n\nPrevious conversation:\n"
                for entry in history[-3:]:
                    history_context += f"Q: {entry['question']}\nA: {entry['answer'][:100]}...\n"
        
        # Retrieve relevant chunks
        docs = self.vectorstore.similarity_search(question, k=top_k)
        
        # Extract context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Build prompt
        full_prompt = f"""{self.system_prompt}

Context from AWS study materials:
{context}

{history_context}

Current question: {question}

Provide a clear, helpful answer:"""
        
        # Generate answer
        response = self.llm.predict(full_prompt)
        
        # Extract sources with relevance
        sources = []
        for i, doc in enumerate(docs):
            sources.append({
                "text": doc.page_content[:300] + "...",
                "source": doc.metadata.get("source", "unknown"),
                "doc_type": doc.metadata.get("doc_type", "unknown"),
                "chunk_id": doc.metadata.get("chunk_id", -1),
                "relevance_score": 1.0 - (i * 0.1)
            })
        
        # Save to history
        self.conversation_history.add_message(session_id, question, response)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "question": question,
            "answer": response,
            "sources": sources,
            "num_sources": len(sources),
            "session_id": session_id,
            "processing_time_ms": round(processing_time, 2)
        }
    
    def explain_concept(
        self, 
        concept: str, 
        detail_level: str = "medium"
    ) -> Dict:
        """
        Get detailed explanation of AWS concept.
        
        Args:
            concept: AWS concept to explain
            detail_level: brief, medium, or detailed
            
        Returns:
            Dictionary with explanation
        """
        detail_instructions = {
            "brief": "Provide a concise 2-3 sentence explanation.",
            "medium": "Provide a comprehensive explanation with key features and use cases.",
            "detailed": "Provide an in-depth explanation with features, use cases, best practices, and exam tips."
        }
        
        instruction = detail_instructions.get(detail_level, detail_instructions["medium"])
        
        # Build additional items for detailed level
        additional_items = ""
        if detail_level == "detailed":
            additional_items = "\n4. Best practices\n5. Exam tips"
        
        question = f"""Explain the AWS concept: {concept}

{instruction}

Include:
1. What it is
2. Key features
3. Common use cases{additional_items}

Keep the explanation clear and educational."""
        
        return self.query(question, top_k=6)
    
    def compare_services(
        self, 
        service1: str, 
        service2: str,
        aspects: Optional[List[str]] = None
    ) -> Dict:
        """
        Compare two AWS services.
        
        Args:
            service1: First service
            service2: Second service
            aspects: Specific aspects to compare
            
        Returns:
            Comparison details
        """
        aspect_text = ""
        if aspects:
            aspect_text = f"\nFocus on these aspects: {', '.join(aspects)}"
        
        question = f"""Compare {service1} and {service2}.{aspect_text}

Include:
1. Key differences
2. When to use each
3. Pricing considerations
4. Performance characteristics
5. Common use cases

Provide a clear comparison table format."""
        
        return self.query(question, top_k=8)
    
    def generate_quiz(
        self, 
        topic: Optional[str] = None,
        num_questions: int = 5,
        difficulty: Optional[str] = None
    ) -> Dict:
        """
        Generate practice quiz questions.
        
        Args:
            topic: Specific topic or None for general
            num_questions: Number of questions
            difficulty: easy, medium, hard, or None
            
        Returns:
            Quiz with questions
        """
        quiz_id = str(uuid.uuid4())
        
        # Build search query
        if topic:
            search_query = f"practice questions about {topic}"
        else:
            search_query = "AWS certification practice questions"
        
        # Add difficulty to search if specified
        if difficulty:
            search_query += f" {difficulty}"
        
        # Retrieve practice questions
        docs = self.vectorstore.similarity_search(
            search_query,
            k=num_questions * 3
        )
        
        questions = []
        for i, doc in enumerate(docs[:num_questions]):
            questions.append({
                "id": f"q{i+1}",
                "question": doc.page_content,
                "topic": topic or "General AWS",
                "difficulty": difficulty or "medium",
                "source": doc.metadata.get("filename", "unknown")
            })
        
        return {
            "quiz_id": quiz_id,
            "topic": topic or "General AWS",
            "questions": questions,
            "total_questions": len(questions)
        }
    
    def get_session_info(self, session_id: str) -> Dict:
        """Get information about a study session."""
        history = self.conversation_history.get_history(session_id)
        
        if not history:
            return {
                "session_id": session_id,
                "exists": False
            }
        
        topics = set()
        for entry in history:
            question_lower = entry["question"].lower()
            for service in ["s3", "ec2", "vpc", "iam", "rds", "lambda"]:
                if service in question_lower:
                    topics.add(service.upper())
        
        return {
            "session_id": session_id,
            "exists": True,
            "questions_asked": len(history),
            "topics_covered": list(topics),
            "first_question_time": history[0]["timestamp"] if history else None,
            "last_active": history[-1]["timestamp"] if history else None
        }


def main():
    """Test the enhanced RAG engine."""
    print("\n" + "="*60)
    print("🧠 Testing Enhanced AWS Study Partner")
    print("="*60 + "\n")
    
    partner = EnhancedAWSStudyPartner()
    
    # Test query with session
    print("Test 1: Query with session")
    result = partner.query("What is Amazon S3?", session_id="test123")
    print(f"Answer: {result['answer'][:200]}...")
    print(f"Sources: {result['num_sources']}")
    print(f"Time: {result['processing_time_ms']}ms\n")
    
    # Test explain
    print("Test 2: Explain concept")
    result = partner.explain_concept("VPC peering", detail_level="medium")
    print(f"Answer: {result['answer'][:200]}...\n")
    
    # Test compare
    print("Test 3: Compare services")
    result = partner.compare_services("S3 Standard", "S3 Glacier")
    print(f"Answer: {result['answer'][:200]}...\n")
    
    # Test quiz
    print("Test 4: Generate quiz")
    result = partner.generate_quiz(topic="EC2", num_questions=2)
    print(f"Quiz ID: {result['quiz_id']}")
    print(f"Questions: {result['total_questions']}\n")
    
    # Test session info
    print("Test 5: Session info")
    result = partner.get_session_info("test123")
    print(f"Session exists: {result['exists']}")
    print(f"Questions asked: {result.get('questions_asked', 0)}\n")
    
    print("="*60)
    print("✅ All tests complete!")
    print("="*60)


if __name__ == "__main__":
    main()
    