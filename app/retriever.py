from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from app.config import *

embedder = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GEMINI_API_KEY
)

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def retrieve(query: str, top_k: int = TOP_K_RESULTS, source_filter: str = None) -> list:
    query_vector = embedder.embed_query(query)

    search_filter = None
    if source_filter:
        search_filter = Filter(must=[
            FieldCondition(
                key="source",
                match=MatchValue(value=source_filter)
            )
        ])

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        query_filter=search_filter,
        with_payload=True,
        score_threshold=0.4
    )

    return [
        {
            "text": r.payload["text"],
            "source": r.payload["source"],
            "score": round(r.score, 3)
        }
        for r in results
    ]