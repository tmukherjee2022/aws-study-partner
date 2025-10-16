"""Minimal OpenAI test - bypassing any langchain issues."""
import os
from dotenv import load_dotenv

load_dotenv()

print("Testing OpenAI API...\n")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ No API key found")
    exit(1)

print(f"API Key: {api_key[:20]}...{api_key[-4:]}\n")

# Import OpenAI
import openai
print(f"OpenAI version: {openai.__version__}\n")

# Test with direct API call (no client object)
try:
    print("Testing embedding generation...")
    
    # Use the module-level API (works with 1.12.0)
    openai.api_key = api_key
    
    # Create embedding using the direct method
    from openai import OpenAI
    client = OpenAI()  # Will use openai.api_key automatically
    
    response = client.embeddings.create(
        input="test",
        model="text-embedding-3-large"
    )
    
    print(f"✅ OpenAI API working!")
    print(f"   Embedding dimension: {len(response.data[0].embedding)}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()