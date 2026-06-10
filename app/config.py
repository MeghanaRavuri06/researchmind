from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
QDRANT_HOST      = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT      = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME  = "researchmind"
EMBEDDING_MODEL  = "models/gemini-embedding-001"
LLM_MODEL        = "models/gemini-2.5-flash-lite"
CHUNK_SIZE       = 800
CHUNK_OVERLAP    = 150
TOP_K_RESULTS    = 5