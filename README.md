# 🔬 ResearchMind

> An AI-powered research assistant that lets you upload research papers and ask questions about them — built with RAG (Retrieval-Augmented Generation).

🔗 **Live Demo:** [http://13.235.42.50:8501](http://13.235.42.50:8501)

---

## What It Does

ResearchMind lets you interact with research papers and documents using natural language. Upload a PDF or paste a URL, ask any question, and get accurate answers grounded in the document — with source citations.

**Example:**
> Upload the RAG paper → Ask *"What problem does RAG solve?"* → Get a cited, accurate answer instantly.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend API | FastAPI |
| Vector Database | Qdrant |
| Embeddings | Gemini Embedding (`gemini-embedding-001`) |
| LLM | Gemini 2.5 Flash Lite |
| Orchestration | LangGraph |
| Containerization | Docker & Docker Compose |
| Cloud | AWS EC2 |

---

## Architecture

```
User (Browser)
      │
      ▼
 Streamlit Frontend  (port 8501)
      │
      ▼
 FastAPI Backend     (port 8000)
      │
      ├──► Gemini Embeddings API  ──► Qdrant Vector DB  (port 6333)
      │                                      │
      │                               Vector Search
      │                                      │
      └──► LangGraph Agent  ◄────── Retrieved Chunks
                │
                ▼
          Gemini LLM
                │
                ▼
         Answer + Citations
```

---

## Key Features

- **PDF & URL ingestion** — upload documents or paste any URL to ingest content
- **Semantic search** — chunks are embedded and stored in Qdrant for fast similarity search
- **RAG pipeline** — retrieved context is passed to Gemini LLM for grounded answers
- **Source citations** — every answer references the source document
- **LangGraph agent** — orchestrates multi-step retrieval and generation
- **Dockerized** — all services run in isolated containers via Docker Compose
- **Cloud deployed** — live on AWS EC2 with proper networking and security groups

---

## Running Locally

### Prerequisites
- Docker & Docker Compose
- Gemini API key ([get one free here](https://aistudio.google.com))

### Setup

```bash
# Clone the repo
git clone https://github.com/MeghanaRavuri06/researchmind.git
cd researchmind

# Add your API key
echo "GEMINI_API_KEY=your_key_here" > .env

# Start all services
docker-compose up --build
```

Then open [http://localhost:8501](http://localhost:8501)

### Services
| Service | URL |
|---|---|
| Streamlit Frontend | http://localhost:8501 |
| FastAPI Backend | http://localhost:8000/docs |
| Qdrant Dashboard | http://localhost:6333/dashboard |

---

## Project Structure

```
researchmind/
├── app/
│   ├── main.py          # FastAPI routes
│   ├── ingest.py        # PDF/URL processing & embedding
│   ├── retriever.py     # Qdrant vector search
│   ├── agent.py         # LangGraph RAG agent
│   └── config.py        # Configuration & constants
├── frontend/
│   └── app.py           # Streamlit UI
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## How It Works

1. **Ingestion** — PDF text is extracted, split into 800-token chunks with 150-token overlap, embedded using Gemini, and stored in Qdrant
2. **Retrieval** — user query is embedded and compared against stored vectors using cosine similarity; top 5 chunks are retrieved
3. **Generation** — LangGraph agent passes retrieved chunks as context to Gemini LLM, which generates a cited answer
4. **Response** — answer and source references are returned to the Streamlit UI

---

## Deployment

Deployed on **AWS EC2** using Docker Compose:
- EC2 Security Groups configured for ports 8501 (frontend) and 8000 (API)
- All services communicate over a Docker bridge network
- Qdrant data persisted via Docker volume

---

## Author

**Meghana Ravuri**
[GitHub](https://github.com/MeghanaRavuri06)

