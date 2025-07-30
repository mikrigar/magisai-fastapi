#!/usr/bin/env python3
"""
MagisAI FastAPI Gateway
Connects Weaviate → RunPod Mistral-Nemo → Response
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import weaviate
from weaviate.classes.init import Auth
import httpx
import logging
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="MagisAI Gateway", version="1.0.0")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from .env file
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_KEY = os.getenv("WEAVIATE_KEY")
RUNPOD_URL = os.getenv("RUNPOD_URL")
RUNPOD_KEY = os.getenv("RUNPOD_KEY")

# Debug: Print config (remove API keys from logs)
logger.info(f"🔧 Weaviate URL: {WEAVIATE_URL}")
logger.info(f"🔧 Weaviate Key: {'***' + WEAVIATE_KEY[-4:] if WEAVIATE_KEY else 'NOT SET'}")
logger.info(f"🔧 RunPod URL: {RUNPOD_URL}")
logger.info(f"🔧 RunPod Key: {'***' + RUNPOD_KEY[-4:] if RUNPOD_KEY else 'NOT SET'}")

# Initialize Weaviate client
try:
    if not WEAVIATE_URL or not WEAVIATE_KEY:
        raise ValueError("WEAVIATE_URL and WEAVIATE_KEY must be set in .env file")
        
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
    max_chunks: int = 5
    agent_filter: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    confidence: float
    agent_used: str

class HealthResponse(BaseModel):
    status: str
    weaviate_connected: bool
    runpod_available: bool

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint - simplified for debugging"""
    logger.info("🏥 Health check requested")
    
    # Quick Weaviate check without hanging
    weaviate_ok = False
    try:
        weaviate_ok = weaviate_client is not None and weaviate_client.is_ready()
        logger.info(f"Weaviate status: {weaviate_ok}")
    except Exception as e:
        logger.error(f"Weaviate check failed: {e}")
        weaviate_ok = False
    
    # Skip RunPod check for now to avoid timeouts
    runpod_ok = bool(RUNPOD_URL and RUNPOD_KEY)
    logger.info(f"RunPod configured: {runpod_ok}")
    
    result = HealthResponse(
        status="healthy" if weaviate_ok else "degraded",
        weaviate_connected=weaviate_ok,
        runpod_available=runpod_ok
    )
    
    logger.info(f"Health check result: {result}")
    return result

def detect_theological_intent(question: str) -> Dict[str, any]:
    """Detect which theological domain(s) the question relates to"""
    question_lower = question.lower()
    
    intent_keywords = {
        "Suffering_and_Evil": ["suffering", "evil", "pain", "why god allow", "problem of evil"],
        "Evidence_of_Gods_Existence": ["god exist", "proof", "evidence", "atheism", "creation"],
        "Christian_Theology": ["trinity", "jesus", "christ", "incarnation", "salvation"],
        "Happiness_and_Suffering": ["happiness", "joy", "levels", "purpose", "meaning"],
        "Prayer_and_Spiritual_Life": ["prayer", "spiritual", "contemplation", "mysticism"],
        "Moral_and_Social_Teaching": ["ethics", "moral", "conscience", "social", "teaching"],
        "Evidence_of_life_after_death_and_a_soul": ["soul", "afterlife", "death", "consciousness"],
        "Science_and_Faith": ["science", "evolution", "big bang", "faith", "reason"]
    }
    
    detected_agents = []
    confidence_scores = {}
    
    for agent, keywords in intent_keywords.items():
        score = sum(1 for keyword in keywords if keyword in question_lower)
        if score > 0:
            detected_agents.append(agent)
            confidence_scores[agent] = score
    
    primary_agent = max(confidence_scores, key=confidence_scores.get) if confidence_scores else "Christian_Theology"
    
    return {
        "primary_agent": primary_agent,
        "all_agents": detected_agents,
        "confidence": confidence_scores.get(primary_agent, 0.1)
    }

def retrieve_context(question: str, max_chunks: int = 5, agent_filter: str = None) -> List[Dict]:
    """Retrieve relevant chunks from Weaviate"""
    if not weaviate_client:
        raise HTTPException(status_code=503, detail="Weaviate not connected")
    
    try:
        collection = weaviate_client.collections.get("MagisChunk")
        
        # Simple search without filtering for now (we'll add filtering back later)
        response = collection.query.near_text(
            query=question,
            limit=max_chunks,
            return_metadata=["distance"]
        )
        
        chunks = []
        for obj in response.objects:
            chunks.append({
                "content": obj.properties.get("content", ""),
                "agent": obj.properties.get("agent1", "Unknown"),
                "topics": obj.properties.get("agent1CoreTopics", []),
                "source": obj.properties.get("sourceFile", "Unknown"),
                "human_id": obj.properties.get("humanId", ""),
                "distance": obj.metadata.distance if hasattr(obj.metadata, 'distance') else 1.0
            })
        
        logger.info(f"✅ Retrieved {len(chunks)} chunks for query: {question[:50]}...")
        return chunks
        
    except Exception as e:
        logger.error(f"❌ Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

async def generate_answer(question: str, context_chunks: List[Dict]) -> str:
    """Generate answer using RunPod Mistral-Nemo"""
    
    # Build context prompt
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(
            f"[Source {i} - {chunk['agent']}: {chunk['source']}]\n{chunk['content']}\n"
        )
    
    context = "\n".join(context_parts)
    
    # Mistral-Nemo optimized prompt
    prompt = f"""<s>[INST] You are MagisAI, a Catholic theology assistant. Answer the user's question accurately and concisely using the authoritative sources provided.

Context Sources:
{context}

User Question: {question}

Provide a clear, theologically sound answer based on the sources. Be concise but thorough. If the sources don't contain relevant information, say so honestly. [/INST]"""

    try:
        if not RUNPOD_URL or not RUNPOD_KEY:
            raise ValueError("RunPod credentials not configured")
            
        async with httpx.AsyncClient() as client:
            # RunPod Mistral-Nemo payload for chat completions
            payload = {
                "model": "mistralai/Mistral-Nemo-Instruct-2407",
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            response = await client.post(
                RUNPOD_URL,  # Use the full URL as-is
                json=payload,
                headers={"Authorization": f"Bearer {RUNPOD_KEY}"},
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                # Extract from OpenAI-style response
                choices = result.get("choices", [])
                if choices:
                    answer = choices[0].get("message", {}).get("content", "").strip()
                    return answer
                else:
                    return "No response generated"
            else:
                raise Exception(f"RunPod error: {response.status_code} - {response.text}")
                
    except Exception as e:
        logger.error(f"❌ Mistral generation failed: {e}")
        # Fallback response
        return f"I apologize, but I'm having difficulty generating a response at the moment. However, I found relevant information in {len(context_chunks)} sources related to your question about '{question}'. The sources indicate this is an important theological matter. Please try again shortly."

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Main query processing endpoint"""
    try:
        logger.info(f"🔍 Processing query: {request.question}")
        
        # Detect intent
        intent = detect_theological_intent(request.question)
        agent_filter = request.agent_filter or intent["primary_agent"]
        
        # Retrieve context
        chunks = retrieve_context(
            question=request.question,
            max_chunks=request.max_chunks,
            agent_filter=agent_filter
        )
        
        if not chunks:
            return QueryResponse(
                answer="I apologize, but I couldn't find relevant information to answer your question. Please try rephrasing or asking about a different theological topic.",
                sources=[],
                confidence=0.0,
                agent_used=agent_filter
            )
        
        # Generate answer with Mistral-Nemo
        answer = await generate_answer(request.question, chunks)
        
        # Calculate confidence
        avg_distance = sum(chunk["distance"] for chunk in chunks) / len(chunks)
        confidence = max(0.0, 1.0 - avg_distance)
        
        return QueryResponse(
            answer=answer,
            sources=[{
                "agent": chunk["agent"],
                "source": chunk["source"],
                "human_id": chunk["human_id"],
                "distance": chunk["distance"]
            } for chunk in chunks],
            confidence=confidence,
            agent_used=agent_filter
        )
        
    except Exception as e:
        logger.error(f"❌ Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "MagisAI Gateway",
        "version": "1.0.0",
        "status": "ready",
        "mistral_model": "mistral-nemo",
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))