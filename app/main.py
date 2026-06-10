from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.ingest import ingest, extract_text_from_pdf, extract_text_from_url
from app.retriever import retrieve
from app.agent import run_agent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ResearchMind API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    use_agent: bool = True

class URLRequest(BaseModel):
    url: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted")
    content = await file.read()
    text = extract_text_from_pdf(content)
    if not text.strip():
        raise HTTPException(400, "Could not extract text from PDF")
    chunks = ingest(text, source_name=file.filename)
    return {"message": f"Ingested {chunks} chunks", "source": file.filename}

@app.post("/ingest/url")
async def ingest_url(req: URLRequest):
    text = extract_text_from_url(req.url)
    chunks = ingest(text, source_name=req.url)
    return {"message": f"Ingested {chunks} chunks", "source": req.url}

@app.post("/query")
async def query(req: QueryRequest):
    logger.info(f"Query received: {req.question}")
    if req.use_agent:
        result = await run_agent(req.question)
    else:
        docs = retrieve(req.question)
        result = {"answer": "Direct retrieval mode", "sources": docs}
    return result