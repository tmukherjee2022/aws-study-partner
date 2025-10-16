"""Test that API keys work."""
import os
from dotenv import load_dotenv

load_dotenv()

print("Testing API keys...\n")

# Test OpenAI
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    print(f"✅ OpenAI key found: {openai_key[:20]}...{openai_key[-4:]}")
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        # Test with a simple embedding
        response = client.embeddings.create(
            input="test",
            model="text-embedding-3-large"
        )
        print("✅ OpenAI API working!")
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
else:
    print("❌ OPENAI_API_KEY not found")

print()

# Test Pinecone
pinecone_key = os.getenv("PINECONE_API_KEY")
if pinecone_key:
    print(f"✅ Pinecone key found: {pinecone_key[:20]}...{pinecone_key[-4:]}")
    
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=pinecone_key)
        indexes = pc.list_indexes()
        print(f"✅ Pinecone API working! Found {len(indexes)} indexes")
    except Exception as e:
        print(f"❌ Pinecone API error: {e}")
else:
    print("❌ PINECONE_API_KEY not found")