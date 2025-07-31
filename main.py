#!/usr/bin/env python3
"""
Enhanced FastAPI RAG Service with Weaviate Query Agent + Intro/Conclusion
Uses Weaviate Query Agent for core answers with Mistral for intro/conclusion OR separate Weaviate agent
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import weaviate
from weaviate.classes.init import Auth
from weaviate.agents.query import QueryAgent
from weaviate_agents.classes import QueryAgentCollectionConfig
import httpx
import logging
import os
import time
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Enhanced RAG API with Query Agent + Intro/Conclusion", version="2.1.0")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_KEY = os.getenv("WEAVIATE_API_KEY") or os.getenv("WEAVIATE_KEY")
RUNPOD_URL = os.getenv("RUNPOD_URL")
RUNPOD_KEY = os.getenv("RUNPOD_KEY")
RUNPOD_MODEL = os.getenv("RUNPOD_MODEL")

# Initialize Weaviate client and agents
weaviate_client = None
main_query_agent = None
intro_conclusion_agent = None  # Optional second agent

try:
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_KEY)
    )
    
    # Main Query Agent for core answers
    main_query_agent = QueryAgent(
        client=weaviate_client,
        collections=[
            QueryAgentCollectionConfig(
                name="MagisChunk",
                target_vector=["content_vector"],
            ),
        ],
    )
    
    # Optional: Second agent for intro/conclusion (if you want pure Weaviate approach)
    # This could use the same collection or a different one optimized for contextual framing
    try:
        intro_conclusion_agent = QueryAgent(
            client=weaviate_client,
            collections=[
                QueryAgentCollectionConfig(
                    name="MagisChunk",  # Could be a different collection for contextual content
                    target_vector=["content_vector"],
                ),
            ],
        )
        logger.info("✅ Both main and intro/conclusion agents initialized")
    except Exception as e:
        logger.warning(f"⚠️ Intro/conclusion agent failed to initialize: {e}")
        intro_conclusion_agent = None
    
    logger.info("✅ Connected to Weaviate with Query Agents")
except Exception as e:
    logger.error(f"❌ Weaviate/Query Agent setup failed: {e}")

class QueryRequest(BaseModel):
    question: str
    user: str
    use_weaviate_intro_conclusion: Optional[bool] = False  # True for Weaviate agent, False for Mistral
    include_intro_conclusion: Optional[bool] = True
    include_debug_info: Optional[bool] = False  # Include raw agent response details
    max_chunks: Optional[int] = 5

class MagisChunkResult(BaseModel):
    content: str
    sourceFile: str
    headerContext: str
    humanId: str
    agent1: str
    agent1CoreTopics: List[str]
    citations: List[str]
    chunkLength: int
    citationCount: int
    object_id: str
    distance: Optional[float] = None

class QueryResponse(BaseModel):
    intro: Optional[str] = None
    answer: str  # Core answer from Weaviate Query Agent (final_answer only)
    conclusion: Optional[str] = None
    magis_chunk_results: List[MagisChunkResult] = []  # Detailed source information
    processing_time: float
    method_used: str
    agent_confidence: Optional[float] = None
    intro_conclusion_method: Optional[str] = None  # "mistral" or "weaviate_agent"
    # Debug info (optional)
    raw_agent_info: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    return {
        "service": "Enhanced RAG API with Query Agent + Intro/Conclusion",
        "version": "2.1.0",
        "features": [
            "weaviate_query_agent", 
            "mistral_intro_conclusion", 
            "optional_weaviate_intro_conclusion_agent",
            "pure_agent_responses"
        ],
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "weaviate_connected": weaviate_client is not None and weaviate_client.is_ready(),
        "main_query_agent_ready": main_query_agent is not None,
        "intro_conclusion_agent_ready": intro_conclusion_agent is not None,
        "mistral_configured": bool(RUNPOD_URL and RUNPOD_KEY),
        "runpod_model": RUNPOD_MODEL,
        "agent_collections": ["MagisChunk"] if main_query_agent else []
    }

async def generate_mistral_intro_conclusion(question: str, core_answer: str) -> Dict[str, str]:
    """Generate intro and conclusion using Mistral"""
    
    if not RUNPOD_URL or not RUNPOD_KEY:
        return {
            "intro": "Here's what I found regarding your question:",
            "conclusion": "I hope this information is helpful to you."
        }
    
    # Generate intro
    intro_prompt = f"""Based on this theological/philosophical question, write a brief, engaging introduction (1-2 sentences) that sets up the answer:

Question: "{question}"

Write only the introduction, nothing else. Make it welcoming and contextual."""

    # Generate conclusion  
    conclusion_prompt = f"""Based on this question and answer, write a brief conclusion (1-2 sentences) that wraps up the response helpfully:

Question: "{question}"
Answer: "{core_answer[:500]}..."

Write only the conclusion, nothing else. Make it encouraging and offer further assistance if needed."""

    intro = "Here's what I found regarding your question:"
    conclusion = "I hope this information helps with your inquiry."
    
    try:
        async with httpx.AsyncClient() as client:
            # Generate intro
            intro_response = await client.post(
                RUNPOD_URL,
                json={
                    "model": RUNPOD_MODEL,
                    "messages": [{"role": "user", "content": intro_prompt}],
                    "max_tokens": 100,
                    "temperature": 0.4,
                    "stream": False
                },
                headers={
                    "Authorization": f"Bearer {RUNPOD_KEY}",
                    "Content-Type": "application/json"
                },
                timeout=15.0
            )
            
            if intro_response.status_code == 200:
                result = intro_response.json()
                if result.get("choices") and result["choices"][0].get("message"):
                    intro = result["choices"][0]["message"]["content"].strip()
                elif result.get("response"):
                    intro = result["response"].strip()
            
            # Generate conclusion
            conclusion_response = await client.post(
                RUNPOD_URL,
                json={
                    "model": RUNPOD_MODEL,
                    "messages": [{"role": "user", "content": conclusion_prompt}],
                    "max_tokens": 100,
                    "temperature": 0.4,
                    "stream": False
                },
                headers={
                    "Authorization": f"Bearer {RUNPOD_KEY}",
                    "Content-Type": "application/json"
                },
                timeout=15.0
            )
            
            if conclusion_response.status_code == 200:
                result = conclusion_response.json()
                if result.get("choices") and result["choices"][0].get("message"):
                    conclusion = result["choices"][0]["message"]["content"].strip()
                elif result.get("response"):
                    conclusion = result["response"].strip()
            
    except Exception as e:
        logger.error(f"Mistral intro/conclusion generation error: {e}")
    
    return {"intro": intro, "conclusion": conclusion}

async def generate_weaviate_intro_conclusion(question: str, core_answer: str) -> Dict[str, str]:
    """Generate intro and conclusion using Weaviate agent"""
    
    if not intro_conclusion_agent:
        return {
            "intro": "Based on the theological sources available:",
            "conclusion": "This information comes from our knowledge base of theological and philosophical texts."
        }
    
    try:
        # Generate contextual intro using the agent
        intro_query = f"Provide a brief introduction for answering this question: {question}"
        intro_result = intro_conclusion_agent.run(intro_query)
        intro = str(intro_result.answer) if hasattr(intro_result, 'answer') else str(intro_result)
        
        # Generate contextual conclusion using the agent
        conclusion_query = f"Provide a brief conclusion for this answer about '{question}': {core_answer[:300]}..."
        conclusion_result = intro_conclusion_agent.run(conclusion_query)
        conclusion = str(conclusion_result.answer) if hasattr(conclusion_result, 'answer') else str(conclusion_result)
        
        # Keep them concise (truncate if too long)
        intro = intro[:300] + "..." if len(intro) > 300 else intro
        conclusion = conclusion[:300] + "..." if len(conclusion) > 300 else conclusion
        
        return {"intro": intro, "conclusion": conclusion}
        
    except Exception as e:
        logger.error(f"Weaviate intro/conclusion generation error: {e}")
        return {
            "intro": "Based on the theological sources available:",
            "conclusion": "This information comes from our knowledge base of theological and philosophical texts."
        }

async def get_detailed_source_content(source_ids: List[str]) -> List[MagisChunkResult]:
    """Fetch detailed content for each source ID from MagisChunk collection"""
    
    if not weaviate_client:
        return []
    
    try:
        collection = weaviate_client.collections.get("MagisChunk")
        detailed_results = []
        
        for source_id in source_ids:
            try:
                # Get the full object by ID
                obj = collection.query.fetch_object_by_id(source_id)
                
                if obj:
                    properties = obj.properties
                    
                    # Parse citations if they exist as a string
                    citations_raw = properties.get("citations", "[]")
                    if isinstance(citations_raw, str):
                        try:
                            # Try to parse as a list representation
                            citations = eval(citations_raw) if citations_raw != "[]" else []
                        except:
                            citations = [citations_raw] if citations_raw else []
                    else:
                        citations = citations_raw if isinstance(citations_raw, list) else []
                    
                    # Parse agent1CoreTopics if it exists as a string
                    core_topics_raw = properties.get("agent1CoreTopics", "[]")
                    if isinstance(core_topics_raw, str):
                        try:
                            core_topics = eval(core_topics_raw) if core_topics_raw != "[]" else []
                        except:
                            core_topics = [core_topics_raw] if core_topics_raw else []
                    else:
                        core_topics = core_topics_raw if isinstance(core_topics_raw, list) else []
                    
                    result = MagisChunkResult(
                        content=properties.get("content", ""),
                        sourceFile=properties.get("sourceFile", "Unknown"),
                        headerContext=properties.get("headerContext", ""),
                        humanId=properties.get("humanId", ""),
                        agent1=properties.get("agent1", ""),
                        agent1CoreTopics=core_topics,
                        citations=citations,
                        chunkLength=properties.get("chunkLength", 0),
                        citationCount=properties.get("citationCount", 0),
                        object_id=source_id,
                        distance=getattr(obj.metadata, 'distance', None) if hasattr(obj, 'metadata') else None
                    )
                    detailed_results.append(result)
                    
            except Exception as e:
                logger.error(f"Failed to fetch details for source {source_id}: {e}")
                continue
        
        logger.info(f"✅ Retrieved detailed content for {len(detailed_results)} sources")
        return detailed_results
        
    except Exception as e:
        logger.error(f"❌ Failed to get detailed source content: {e}")
        return []

async def query_with_agent_plus_framing(question: str, use_weaviate_framing: bool = False, include_framing: bool = True, include_debug: bool = False) -> Dict[str, Any]:
    """Use Weaviate Query Agent for core answer, with optional intro/conclusion framing"""
    
    if not main_query_agent:
        raise HTTPException(status_code=503, detail="Main Query Agent not available")
    
    try:
        # Get core answer from Weaviate Query Agent
        logger.info("🔍 Running main query agent...")
        result = main_query_agent.run(question)
        
        # Extract just the final_answer from the result
        core_answer = ""
        if hasattr(result, 'final_answer'):
            core_answer = str(result.final_answer)
        elif hasattr(result, 'answer'):
            core_answer = str(result.answer)
        else:
            core_answer = str(result)
        
        logger.info(f"📝 Extracted core answer: {len(core_answer)} characters")
        
        # Extract source IDs for detailed lookup
        source_ids = []
        if hasattr(result, 'sources') and result.sources:
            for source in result.sources:
                if hasattr(source, 'object_id'):
                    source_ids.append(source.object_id)
                elif hasattr(source, 'id'):
                    source_ids.append(source.id)
        
        logger.info(f"🔗 Found {len(source_ids)} source IDs")
        
        # Get detailed source content
        detailed_sources = await get_detailed_source_content(source_ids)
        
        # Generate intro and conclusion if requested
        intro = None
        conclusion = None
        framing_method = None
        
        if include_framing:
            if use_weaviate_framing and intro_conclusion_agent:
                logger.info("🎭 Generating intro/conclusion with Weaviate agent...")
                framing = await generate_weaviate_intro_conclusion(question, core_answer)
                framing_method = "weaviate_agent"
            else:
                logger.info("🎭 Generating intro/conclusion with Mistral...")
                framing = await generate_mistral_intro_conclusion(question, core_answer)
                framing_method = "mistral"
            
            intro = framing.get("intro")
            conclusion = framing.get("conclusion")
        
        # Prepare debug info if requested
        raw_agent_info = None
        if include_debug:
            raw_agent_info = {
                "output_type": getattr(result, 'output_type', 'unknown'),
                "original_query": getattr(result, 'original_query', question),
                "collection_names": getattr(result, 'collection_names', []),
                "total_time": getattr(result, 'total_time', 0),
                "usage": str(getattr(result, 'usage', 'N/A')),
                "is_partial_answer": getattr(result, 'is_partial_answer', False),
                "source_count": len(source_ids)
            }
        
        logger.info(f"✅ Query Agent + framing completed successfully")
        return {
            "intro": intro,
            "answer": core_answer,
            "conclusion": conclusion,
            "detailed_sources": detailed_sources,
            "confidence": getattr(result, 'confidence', 0.9),
            "framing_method": framing_method,
            "raw_agent_info": raw_agent_info
        }
        
    except Exception as e:
        logger.error(f"❌ Query Agent + framing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query Agent error: {str(e)}")

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Main query endpoint using Weaviate Query Agent with optional intro/conclusion"""
    
    start_time = time.time()
    logger.info(f"Query from {request.user}: {request.question}")
    
    try:
        # Use Query Agent for core answer with optional framing
        result = await query_with_agent_plus_framing(
            question=request.question,
            use_weaviate_framing=request.use_weaviate_intro_conclusion,
            include_framing=request.include_intro_conclusion,
            include_debug=request.include_debug_info
        )
        
        return QueryResponse(
            intro=result["intro"],
            answer=result["answer"],
            conclusion=result["conclusion"],
            magis_chunk_results=result["detailed_sources"],
            processing_time=time.time() - start_time,
            method_used="weaviate_query_agent_with_detailed_sources",
            agent_confidence=result.get("confidence"),
            intro_conclusion_method=result.get("framing_method"),
            raw_agent_info=result.get("raw_agent_info")
        )
        
    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-pure-agent")
async def ask_pure_agent(request: QueryRequest):
    """Endpoint that uses only Weaviate Query Agent (no intro/conclusion)"""
    
    if not main_query_agent:
        raise HTTPException(status_code=503, detail="Query Agent not available")
    
    # Force no framing
    request.include_intro_conclusion = False
    return await ask_question(request)

@app.post("/ask-with-mistral-framing")
async def ask_with_mistral_framing(request: QueryRequest):
    """Endpoint that forces Mistral for intro/conclusion"""
    
    request.use_weaviate_intro_conclusion = False
    request.include_intro_conclusion = True
    return await ask_question(request)

@app.post("/ask-with-weaviate-framing")
async def ask_with_weaviate_framing(request: QueryRequest):
    """Endpoint that forces Weaviate agent for intro/conclusion"""
    
    if not intro_conclusion_agent:
        raise HTTPException(status_code=503, detail="Intro/conclusion agent not available")
    
    request.use_weaviate_intro_conclusion = True
    request.include_intro_conclusion = True
    return await ask_question(request)

# Legacy endpoints for backward compatibility
@app.post("/ask-agent")
async def ask_with_agent_only(request: QueryRequest):
    """Legacy endpoint - now uses pure Weaviate Query Agent"""
    return await ask_pure_agent(request)

@app.post("/ask-basic")
async def ask_with_basic_search_only(request: QueryRequest):
    """Legacy endpoint - now redirects to main agent endpoint"""
    return await ask_question(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
