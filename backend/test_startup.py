"""Debug API startup to find where it hangs."""
import sys
print("1. Starting imports...")

try:
    print("2. Importing os and time...")
    import os
    import time
    
    print("3. Importing dotenv...")
    from dotenv import load_dotenv
    
    print("4. Loading environment variables...")
    load_dotenv()
    
    print("5. Checking API keys...")
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    
    if openai_key:
        print(f"   ✅ OpenAI key found: {openai_key[:20]}...")
    else:
        print("   ❌ OpenAI key NOT found")
        
    if pinecone_key:
        print(f"   ✅ Pinecone key found: {pinecone_key[:20]}...")
    else:
        print("   ❌ Pinecone key NOT found")
    
    print("6. Importing langchain packages...")
    from langchain_openai import OpenAIEmbeddings
    print("   ✅ OpenAIEmbeddings imported")
    
    from langchain_pinecone import Pinecone as PineconeVectorStore
    print("   ✅ PineconeVectorStore imported")
    
    print("7. Initializing embeddings...")
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    )
    print("   ✅ Embeddings initialized")
    
    print("8. Connecting to Pinecone...")
    index_name = os.getenv("PINECONE_INDEX_NAME", "aws-study-partner")
    print(f"   Looking for index: {index_name}")
    
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    print("   ✅ Pinecone connected!")
    
    print("\n✅ All components initialized successfully!")
    print("The issue is NOT with initialization.")
    
except Exception as e:
    print(f"\n❌ Error at step: {e}")
    import traceback
    traceback.print_exc()