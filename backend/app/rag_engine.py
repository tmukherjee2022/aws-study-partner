"""RAG Query Engine - Retrieves context and generates answers."""
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

load_dotenv()


class AWSStudyPartner:
    """RAG-based AWS Study Partner."""
    
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
            model_name="gpt-3.5-turbo",  # Fast and cheap for studying
            temperature=0.7,  # Slightly creative but focused
            max_tokens=500
        )
        
        # Create custom prompt
        self.prompt_template = """You are an expert AWS certification study partner. Use the following context from AWS certification materials to answer the student's question.

Context from study materials:
{context}

Student's question: {question}

Instructions:
- Provide clear, accurate answers based on the context
- If the context doesn't contain enough information, say so
- Use examples when helpful
- For practice questions, explain why answers are correct/incorrect
- Keep responses concise but thorough

Answer:"""

        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
        print("✅ AWS Study Partner initialized")
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """
        Query the knowledge base and generate an answer.
        
        Args:
            question: The student's question
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Retrieve relevant chunks
        docs = self.vectorstore.similarity_search(question, k=top_k)
        
        # Extract context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate answer using LLM
        formatted_prompt = self.prompt.format(context=context, question=question)
        response = self.llm.predict(formatted_prompt)
        
        # Extract source information
        sources = []
        for doc in docs:
            sources.append({
                "text": doc.page_content[:200] + "...",
                "source": doc.metadata.get("source", "unknown"),
                "doc_type": doc.metadata.get("doc_type", "unknown"),
                "chunk_id": doc.metadata.get("chunk_id", -1)
            })
        
        return {
            "question": question,
            "answer": response,
            "sources": sources,
            "num_sources": len(sources)
        }
    
    def practice_quiz(self, topic: str = None, num_questions: int = 5) -> Dict:
        """
        Generate practice questions on a topic.
        
        Args:
            topic: Specific AWS topic (e.g., "S3", "VPC", "EC2")
            num_questions: Number of questions to generate
            
        Returns:
            Dictionary with generated questions
        """
        if topic:
            query = f"practice questions about {topic}"
        else:
            query = "AWS certification practice questions"
        
        # Retrieve practice questions from materials
        docs = self.vectorstore.similarity_search(
            query, 
            k=num_questions * 2,  # Get more to filter
            filter={"doc_type": "questions"}  # Only practice test chunks
        )
        
        if not docs:
            return {
                "topic": topic,
                "questions": [],
                "message": "No practice questions found. Try a different topic."
            }
        
        questions = []
        for doc in docs[:num_questions]:
            questions.append({
                "content": doc.page_content,
                "source": doc.metadata.get("filename", "unknown")
            })
        
        return {
            "topic": topic or "General AWS",
            "questions": questions,
            "count": len(questions)
        }
    
    def explain_concept(self, concept: str) -> Dict:
        """
        Get a detailed explanation of an AWS concept.
        
        Args:
            concept: AWS concept to explain (e.g., "VPC peering", "S3 versioning")
            
        Returns:
            Dictionary with detailed explanation
        """
        # Custom prompt for explanations
        explanation_prompt = f"""Explain the following AWS concept in detail, as if teaching a student preparing for certification:

Concept: {concept}

Include:
1. What it is
2. Why it's used
3. Key features
4. Common use cases
5. Important exam tips

Use the context provided to give accurate information."""
        
        return self.query(explanation_prompt, top_k=5)
    
    def compare_services(self, service1: str, service2: str) -> Dict:
        """
        Compare two AWS services.
        
        Args:
            service1: First AWS service
            service2: Second AWS service
            
        Returns:
            Dictionary with comparison
        """
        question = f"Compare {service1} and {service2}. What are the key differences, and when would you use each?"
        return self.query(question, top_k=6)


def main():
    """Test the RAG engine with sample queries."""
    print("\n" + "="*60)
    print("🧠 Testing AWS Study Partner RAG Engine")
    print("="*60 + "\n")
    
    # Initialize
    partner = AWSStudyPartner()
    
    # Test queries
    test_queries = [
        {
            "type": "basic",
            "question": "What is Amazon S3 and what are its main features?"
        },
        {
            "type": "concept",
            "concept": "VPC peering"
        },
        {
            "type": "comparison",
            "service1": "S3 Standard",
            "service2": "S3 Glacier"
        },
        {
            "type": "practice",
            "topic": "EC2"
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {test['type'].upper()}")
        print("="*60)
        
        if test["type"] == "basic":
            result = partner.query(test["question"])
            print(f"\n❓ Question: {result['question']}")
            print(f"\n💡 Answer:\n{result['answer']}")
            print(f"\n📚 Used {result['num_sources']} sources")
            
        elif test["type"] == "concept":
            result = partner.explain_concept(test["concept"])
            print(f"\n📖 Explaining: {test['concept']}")
            print(f"\n💡 Explanation:\n{result['answer']}")
            
        elif test["type"] == "comparison":
            result = partner.compare_services(test["service1"], test["service2"])
            print(f"\n⚖️  Comparing: {test['service1']} vs {test['service2']}")
            print(f"\n💡 Comparison:\n{result['answer']}")
            
        elif test["type"] == "practice":
            result = partner.practice_quiz(test["topic"], num_questions=2)
            print(f"\n📝 Practice Questions: {result['topic']}")
            print(f"\n   Found {result['count']} questions")
            for j, q in enumerate(result['questions'], 1):
                print(f"\n   Question {j}:")
                print(f"   {q['content'][:200]}...")
    
    print("\n" + "="*60)
    print("✅ RAG Engine test complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()