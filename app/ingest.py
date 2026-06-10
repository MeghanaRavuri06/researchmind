import pypdf
import httpx
import uuid
import io
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.config import *

embedder = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GEMINI_API_KEY
)

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def ensure_collection():
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=3072,
                distance=Distance.COSINE
            )
        )
        print(f"Collection {COLLECTION_NAME} created")

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_url(url: str) -> str:
    resp = httpx.get(url, timeout=15)
    text = re.sub('<[^>]+>', ' ', resp.text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text(text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)

def ingest(text: str, source_name: str) -> int:
    ensure_collection()
    chunks = chunk_text(text)
    embeddings = embedder.embed_documents(chunks)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={"text": chunk, "source": source_name}
        )
        for chunk, emb in zip(chunks, embeddings)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Ingested {len(points)} chunks from {source_name}")
    return len(points)