# AWS Certification Study Partner

An intelligent, RAG-powered study assistant for AWS certifications. Uses semantic search over your study materials and GPT to provide accurate, contextual answers to your AWS questions.

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Next.js](https://img.shields.io/badge/Next.js-14-black)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## 🎯 Project Overview

This application processes AWS certification study materials (PDFs) and creates an intelligent Q&A system using:
- **Retrieval-Augmented Generation (RAG)** architecture
- **OpenAI embeddings** for semantic search
- **Pinecone** vector database for efficient retrieval
- **FastAPI** backend with REST endpoints
- **Next.js** frontend (in development)

**Current Status:** ✅ Backend complete, Frontend in progress

---

## 📊 System Architecture
```
Study Materials (PDFs)
    ↓
PDF Processor (PyPDF/pdfplumber)
    ↓
Text Chunks (1000 chars, 20% overlap)
    ↓
OpenAI Embeddings (text-embedding-3-large)
    ↓
Pinecone Vector Database
    ↓
RAG Query Engine
    ↓
FastAPI REST API
    ↓
Next.js Frontend
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11
- Node.js 18+ (for frontend)
- OpenAI API key
- Pinecone API key

### Backend Setup
```bash
# Clone repository
git clone <your-repo-url>
cd aws-study-partner

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
cd backend
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your API keys

# Process PDFs
python app/pdf_processor.py

# Upload to vector database
python app/vector_store.py

# Start API server
python app/api.py
```

### Frontend Setup (Coming Soon)
```bash
cd frontend
npm install
npm run dev
```

---

## 📁 Project Structure
```
aws-study-partner/
├── backend/
│   ├── app/
│   │   ├── pdf_processor.py      # PDF → text chunks
│   │   ├── vector_store.py       # Chunks → Pinecone
│   │   ├── rag_engine.py         # Query + retrieval logic
│   │   ├── api.py                # FastAPI REST endpoints
│   │   └── cli_study.py          # Command-line interface
│   ├── data/
│   │   ├── raw/                  # Your PDF files (gitignored)
│   │   └── processed/            # JSON chunks
│   ├── .env                      # API keys (gitignored)
│   └── requirements.txt
├── frontend/                     # Next.js app (in progress)
├── .gitignore
└── README.md
```

---

## 🔧 Core Components

### 1. **pdf_processor.py**

**Purpose:** Extracts text from PDFs and splits into semantic chunks.

**Key Functions:**
- `extract_text_from_pdf()` - Extracts text using pdfplumber
- `chunk_text()` - Splits text into 1000-char chunks with 20% overlap
- `process_main_guide()` - Processes certification guide PDFs
- `process_practice_test()` - Processes practice exam PDFs

**Key Design Decisions:**
- **Chunk size: 1000 characters**
  - Large enough for complete concepts
  - Small enough for focused retrieval
  - Optimal for embedding models
- **20% overlap**
  - Prevents splitting important context at boundaries
  - Ensures concepts aren't lost between chunks
- **Metadata tagging**
  - Distinguishes study guides from practice tests
  - Enables filtered searches
  
**Opportunities for Improvement:**
- [ ] Implement smart chunking based on document structure (headers, sections)
- [ ] Add PDF table extraction for pricing/comparison tables
- [ ] Detect and preserve code blocks
- [ ] Handle multi-column layouts better
- [ ] Add chunk quality scoring (remove low-information chunks)

---

### 2. **vector_store.py**

**Purpose:** Generates embeddings and uploads to Pinecone vector database.

**Key Functions:**
- `create_index()` - Creates Pinecone index (3072 dimensions)
- `upload_documents()` - Batch uploads with progress tracking
- `clean_text()` - Removes special tokens that break tokenizer
- `get_vectorstore()` - Retrieves existing index

**Key Design Decisions:**
- **Embedding model: text-embedding-3-large**
  - 3072 dimensions for high semantic accuracy
  - Better technical concept understanding
  - Cost: ~$0.13 per 1M tokens
- **Batch size: 50 chunks**
  - Balances speed vs rate limits
  - Shows progress for long uploads
- **Text cleaning**
  - Removes `<|endoftext|>` and similar tokens
  - Critical for preventing tokenizer errors

**Opportunities for Improvement:**
- [ ] Implement retry logic for rate limit errors
- [ ] Add resume capability if upload fails mid-way
- [ ] Cache embeddings to avoid regeneration
- [ ] Add duplicate detection before upload
- [ ] Implement incremental updates (add new PDFs without re-uploading all)
- [ ] Add embedding quality validation

---

### 3. **rag_engine.py**

**Purpose:** Core RAG logic - retrieves relevant chunks and generates answers.

**Key Functions:**
- `query()` - General Q&A
- `explain_concept()` - Detailed explanations
- `compare_services()` - Service comparisons
- `practice_quiz()` - Retrieves practice questions

**Key Design Decisions:**
- **Retrieval: Top-5 chunks**
  - Provides enough context without overwhelming LLM
  - Balances relevance vs diversity
- **LLM: GPT-3.5-turbo**
  - Fast and cost-effective
  - Sufficient for study Q&A
  - Can upgrade to GPT-4 for complex reasoning
- **Custom prompts**
  - Tailored for study/teaching context
  - Encourages clear, educational responses

**Opportunities for Improvement:**
- [ ] Add conversation memory for follow-up questions
- [ ] Implement re-ranking of retrieved chunks
- [ ] Add source citation in answers
- [ ] Create different prompt templates for different study modes
- [ ] Add confidence scoring for answers
- [ ] Implement query expansion for better retrieval
- [ ] Add spaced repetition algorithm
- [ ] Track which topics user struggles with
- [ ] Generate personalized study plans

---

### 4. **api.py**

**Purpose:** FastAPI REST endpoints for frontend integration.

**Endpoints:**
- `POST /api/query` - Ask questions
- `POST /api/explain` - Get concept explanations
- `POST /api/compare` - Compare services
- `POST /api/quiz` - Get practice questions
- `GET /api/topics` - List available topics
- `GET /health` - Health check

**Key Design Decisions:**
- **Pydantic models** for request/response validation
- **CORS enabled** for Next.js frontend
- **Singleton pattern** for study_partner (avoid reinitialization)
- **Auto-generated docs** at `/docs`

**Opportunities for Improvement:**
- [ ] Add authentication/API keys
- [ ] Implement rate limiting
- [ ] Add caching layer (Redis)
- [ ] Track usage analytics
- [ ] Add WebSocket support for streaming responses
- [ ] Implement user sessions and history
- [ ] Add error tracking (Sentry)

---

## 📚 Key Learnings

### 1. **Document Processing**
- **Text cleaning is critical** - Special tokens can break entire pipeline
- **Chunk overlap prevents information loss** - Without it, concepts get split
- **Metadata matters** - Enables powerful filtering in retrieval

### 2. **Embeddings & Vector Storage**
- **Batch processing saves time** - But watch for rate limits
- **Larger embeddings ≠ always better** - Balance quality vs cost
- **Text normalization** - Remove special characters before embedding

### 3. **RAG Architecture**
- **Retrieval quality > LLM quality** - Good chunks = good answers
- **Prompt engineering matters** - Customize for your use case
- **Top-K selection** - 5 chunks is the sweet spot for most queries

### 4. **Dependency Management**
- **Version conflicts are real** - Pin compatible versions
- **Test imports early** - Catch conflicts before they cascade
- **Virtual environments are essential** - Never skip this step

### 5. **Cost Optimization**
- **Embeddings are one-time costs** - ~$0.14 for 7000 pages
- **Query costs are ongoing** - ~$0.001-0.003 per question
- **Pinecone free tier is generous** - 2GB storage is plenty

---

## 💰 Cost Breakdown

### One-Time Costs (Already Paid)
- **PDF Processing:** Free (local computation)
- **Embedding Generation:** ~$0.14 (4,449 chunks × $0.13/1M tokens)
- **Total:** ~$0.14

### Ongoing Costs
- **Pinecone Storage:** $0/month (free tier)
- **Per Query:** ~$0.001-0.003
  - Embedding query: ~$0.0003
  - LLM response: ~$0.001-0.003
- **100 questions/day:** ~$0.10-0.30/day

---

## 🎯 Roadmap

### Phase 1: Core Backend ✅
- [x] PDF processing
- [x] Vector database setup
- [x] RAG query engine
- [x] FastAPI REST API
- [x] CLI interface

### Phase 2: Frontend (In Progress)
- [ ] Next.js setup
- [ ] Chat interface
- [ ] Quiz mode
- [ ] Topic browser
- [ ] Source citations

### Phase 3: Enhanced Features
- [ ] Conversation history
- [ ] Spaced repetition
- [ ] Progress tracking
- [ ] Flashcard mode
- [ ] Exam simulator
- [ ] Study analytics

### Phase 4: Deployment
- [ ] Docker containerization
- [ ] AWS deployment (ECS/Lambda)
- [ ] CI/CD pipeline
- [ ] Monitoring & logging

---

## 🛠️ Technologies Used

**Backend:**
- Python 3.11
- FastAPI - REST API framework
- LangChain - RAG orchestration
- OpenAI API - Embeddings & LLM
- Pinecone - Vector database
- pdfplumber - PDF text extraction

**Frontend:**
- Next.js 14 - React framework
- TypeScript - Type safety
- Tailwind CSS - Styling
- shadcn/ui - UI components

**Infrastructure:**
- Git - Version control
- Docker - Containerization (coming soon)
- AWS - Deployment (coming soon)

---

## 🔒 Security Notes

- **Never commit `.env` files** - API keys must stay private
- **API keys cost money** - Keep them secure
- **Rate limiting recommended** - Prevent abuse in production
- **Authentication needed** - Add before public deployment

---

## 📝 Environment Variables

Create `backend/.env`:
```bash
# OpenAI
OPENAI_API_KEY=sk-proj-...

# Pinecone
PINECONE_API_KEY=pcsk_...
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=aws-study-partner

# Model Configuration
EMBEDDING_MODEL=text-embedding-3-large
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

---

## 🧪 Testing
```bash
# Test PDF processing
python app/pdf_processor.py

# Test vector store
python app/vector_store.py

# Test RAG engine
python app/rag_engine.py

# Test CLI
python app/cli_study.py

# Test API
python app/api.py
# Visit http://localhost:8000/docs
```

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
1. Better chunk strategies
2. Query result re-ranking
3. Additional study modes
4. Performance optimizations
5. UI/UX enhancements

---

## 📄 License

MIT License - see LICENSE file

---

## 🙏 Acknowledgments

- OpenAI for embeddings and GPT
- Pinecone for vector database
- LangChain for RAG framework
- AWS for certification materials

---

## 📧 Contact

Questions or suggestions? Open an issue!

---

**Built with ❤️ for AWS certification success**
