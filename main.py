#!/usr/bin/env python3
"""
Simplified FastAPI RAG Service
Basic vector search and response generation
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import weaviate
from weaviate.classes.init import Auth
import httpx
import logging
import os
import time
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Simple RAG API", version="1.0.0")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_KEY = os.getenv("WEAVIATE_KEY")
RUNPOD_URL = os.getenv("RUNPOD_URL")
RUNPOD_KEY = os.getenv("RUNPOD_KEY")

# Initialize Weaviate client
try:
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_KEY)
    )
    logger.info("✅ Connected to Weaviate")
except Exception as e:
    logger.error(f"❌ Weaviate connection failed: {e}")
    weaviate_client = None

class QueryRequest(BaseModel):
    question: str
    user: str
    max_chunks: Optional[int] = 5

class Source(BaseModel):
    content: str
    source: str
    distance: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    processing_time: float

@app.get("/")
async def root():
    return {
        "service": "Simple RAG API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "weaviate_connected": weaviate_client is not None and weaviate_client.is_ready(),
        "runpod_configured": bool(RUNPOD_URL and RUNPOD_KEY)
    }

async def search_vectors(question: str, max_chunks: int = 5) -> List[Source]:
    """Search Weaviate for relevant content"""
    
    if not weaviate_client:
        raise HTTPException(status_code=503, detail="Weaviate not connected")
    
    try:
        collection = weaviate_client.collections.get("MagisChunk")
        
        response = collection.query.near_text(
            query=question,
            limit=max_chunks,
            return_metadata=["distance"],
            return_properties=["content", "sourceFile"]
        )
        
        sources = []
        for obj in response.objects:
            source = Source(
                content=obj.properties.get("content", ""),
                source=obj.properties.get("sourceFile", "Unknown"),
                distance=obj.metadata.distance if hasattr(obj.metadata, 'distance') else 1.0
            )
            sources.append(source)
        
        logger.info(f"Retrieved {len(sources)} sources")
        return sources
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

async def generate_answer(question: str, sources: List[Source]) -> str:
    """Generate answer using LLM"""
    
    if not sources:
        return "No relevant information found to answer your question."
    
    # Build context from sources
    context_parts = []
    for i, source in enumerate(sources, 1):
        context_parts.append(f"[{i}] {source.content}")
    
    context = "\n\n".join(context_parts)
    
    # Simple prompt
    prompt = f"""Question: {question}

Context:
{context}

Please provide a helpful answer based on the context above."""

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                RUNPOD_URL,
                json={
                    "model": "mistralai/Mistral-Small-Instruct-2409",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 800,
                    "temperature": 0.3,
                    "stream": False
                },
                headers={"Authorization": f"Bearer {RUNPOD_KEY}"},
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("choices") and result["choices"][0].get("message"):
                    return result["choices"][0]["message"]["content"].strip()
                elif result.get("response"):
                    return result["response"].strip()
            
            raise Exception(f"LLM API error: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return f"Based on the available sources, this question relates to {sources[0].source}. Please try again for a complete response."

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Main query endpoint"""
    
    start_time = time.time()
    logger.info(f"Query from {request.user}: {request.question}")
    
    try:
        # Search for relevant content
        sources = await search_vectors(request.question, request.max_chunks)
        
        # Generate answer
        answer = await generate_answer(request.question, sources)
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
