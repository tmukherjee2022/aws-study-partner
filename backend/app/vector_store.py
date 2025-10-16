import os
from typing import List, Dict
from dotenv import load_dotenv
import json
from pathlib import Path
import time
import re

# Set OpenAI API key BEFORE importing langchain
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore
from pinecone import Pinecone as PineconeClient, ServerlessSpec


def clean_text(text: str) -> str:
    """Remove special tokens and problematic characters from text."""
    # Remove common special tokens
    special_tokens = [
        '<|endoftext|>',
        '<|startoftext|>',
        '<|im_start|>',
        '<|im_end|>',
        '<|system|>',
        '<|user|>',
        '<|assistant|>'
    ]
    
    cleaned = text
    for token in special_tokens:
        cleaned = cleaned.replace(token, '')
    
    # Remove any remaining <|...|> patterns
    cleaned = re.sub(r'<\|[^>]*\|>', '', cleaned)
    
    # Remove extra whitespace
    cleaned = ' '.join(cleaned.split())
    
    return cleaned


class VectorStoreManager:
    def __init__(self):
        # Check for API keys
        openai_key = os.getenv("OPENAI_API_KEY")
        pinecone_key = os.getenv("PINECONE_API_KEY")
        
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        if not pinecone_key:
            raise ValueError("PINECONE_API_KEY not found in .env file")
        
        print(f"✅ OpenAI API key found: {openai_key[:20]}...")
        print(f"✅ Pinecone API key found: {pinecone_key[:20]}...")
        
        # Initialize embeddings
        try:
            self.embeddings = OpenAIEmbeddings(
                model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
            )
            print("✅ OpenAI Embeddings initialized")
        except Exception as e:
            print(f"❌ Error initializing embeddings: {e}")
            raise
        
        # Initialize Pinecone
        self.pc = PineconeClient(api_key=pinecone_key)
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "aws-study-partner")
        
    def create_index(self, dimension: int = 3072):
        """Create Pinecone index if it doesn't exist."""
        print("\nChecking for existing indexes...")
        
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            print(f"Existing indexes: {existing_indexes if existing_indexes else 'None'}")
        except Exception as e:
            print(f"Error listing indexes: {e}")
            existing_indexes = []
        
        if self.index_name not in existing_indexes:
            print(f"\nCreating new index: {self.index_name}")
            try:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
                    )
                )
                print("✅ Index created successfully!")
                print("⏳ Waiting 30 seconds for index to initialize...")
                time.sleep(30)
            except Exception as e:
                print(f"❌ Error creating index: {e}")
                raise
        else:
            print(f"✅ Index '{self.index_name}' already exists")
    
    def load_chunks_from_file(self, file_path: str) -> List[Dict]:
        """Load processed chunks from JSON file."""
        print(f"\nLoading chunks from {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            print(f"✅ Loaded {len(chunks)} chunks")
            return chunks
        except Exception as e:
            print(f"❌ Error loading chunks: {e}")
            raise
    
    def upload_documents(self, chunks: List[Dict], batch_size: int = 50, start_from: int = 0):
        """Upload document chunks to Pinecone in batches."""
        total_chunks = len(chunks)
        print(f"\n{'='*60}")
        print(f"📤 Starting upload of {total_chunks} documents")
        if start_from > 0:
            print(f"⚠️  Resuming from chunk {start_from + 1}")
        print(f"{'='*60}")
        print(f"⏱️  Estimated time: {(total_chunks - start_from) * 0.2 / 60:.1f} minutes")
        print(f"💰 Estimated cost: ${(total_chunks - start_from) * 250 / 1_000_000 * 0.13:.4f}")
        print(f"{'='*60}\n")
        
        # Clean all texts first
        print("🧹 Cleaning text data...")
        texts = [clean_text(chunk["text"]) for chunk in chunks]
        metadatas = [
            {
                **chunk["metadata"],
                "chunk_id": chunk["chunk_id"]
            }
            for chunk in chunks
        ]
        print("✅ Text cleaning complete\n")
        
        # Upload in batches
        total_batches = (len(texts) + batch_size - 1) // batch_size
        vectorstore = None
        
        # Skip to start_from if resuming
        start_batch = start_from // batch_size
        
        for i in range(start_from, len(texts), batch_size):
            batch_num = i // batch_size + 1
            end_idx = min(i + batch_size, len(texts))
            
            print(f"📦 Batch {batch_num}/{total_batches}: "
                  f"chunks {i+1}-{end_idx}...", end=" ", flush=True)
            
            batch_texts = texts[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            
            try:
                if vectorstore is None and i == 0:
                    # First batch - create vectorstore
                    vectorstore = PineconeVectorStore.from_texts(
                        texts=batch_texts,
                        embedding=self.embeddings,
                        metadatas=batch_metadatas,
                        index_name=self.index_name
                    )
                else:
                    # Get existing vectorstore if resuming
                    if vectorstore is None:
                        vectorstore = self.get_vectorstore()
                    
                    # Add texts to existing vectorstore
                    vectorstore.add_texts(
                        texts=batch_texts,
                        metadatas=batch_metadatas
                    )
                print("✅")
            except Exception as e:
                print(f"❌")
                print(f"\n⚠️  Error in batch {batch_num} (chunks {i+1}-{end_idx}): {e}")
                print(f"\nTo resume from this point, run:")
                print(f"  manager.upload_documents(chunks, batch_size=50, start_from={i})")
                raise
        
        print(f"\n{'='*60}")
        print("✅ All documents uploaded successfully!")
        print(f"{'='*60}\n")
        return vectorstore
    
    def get_vectorstore(self):
        """Get existing vector store."""
        return PineconeVectorStore.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings
        )
    
    def test_search(self, query: str, top_k: int = 2):
        """Test the vector store with a sample query."""
        print(f"\n🔍 Query: '{query}'")
        
        try:
            vectorstore = self.get_vectorstore()
            results = vectorstore.similarity_search(query, k=top_k)
            
            for i, doc in enumerate(results, 1):
                print(f"\n📄 Result {i}:")
                print(f"   {doc.page_content[:150]}...")
                print(f"   [Source: {doc.metadata.get('doc_type', 'unknown')}]")
            print()
            return results
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []


def main():
    """Load chunks and upload to vector database."""
    print("\n" + "="*60)
    print("🚀 AWS Study Partner - Vector Store Setup")
    print("="*60)
    
    try:
        manager = VectorStoreManager()
    except Exception as e:
        print(f"\n❌ Initialization error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create index
    try:
        manager.create_index()
    except Exception as e:
        print(f"❌ Failed to create index: {e}")
        return
    
    # Load chunks
    chunks_file = Path("data/processed/all_chunks.json")
    if not chunks_file.exists():
        print(f"\n❌ No processed chunks found at:")
        print(f"   {chunks_file.absolute()}")
        return
    
    try:
        chunks = manager.load_chunks_from_file(str(chunks_file))
    except Exception as e:
        print(f"❌ Failed to load chunks: {e}")
        return
    
    # Upload - start from beginning (change start_from if resuming)
    try:
        manager.upload_documents(chunks, batch_size=50, start_from=0)
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print("\nThe first 18 batches (900 chunks) were already uploaded.")
        print("You can resume from where it failed.")
        return
    
    # Test
    print("\n🧪 Testing with sample queries...\n")
    manager.test_search("What is Amazon S3?")
    manager.test_search("Explain VPC peering")
    
    print("="*60)
    print("✅ Vector database setup complete!")
    print("="*60)


if __name__ == "__main__":
    main()