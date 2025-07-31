#!/usr/bin/env python3
"""
Pure Weaviate Query Agent RAG Service - FIXED VERSION
Uses only Weaviate Query Agents for all functionality - no external LLM dependencies
Fixed: Added missing get_detailed_source_content function
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import weaviate
from weaviate.classes.init import Auth
from weaviate.agents.query import QueryAgent
from weaviate_agents.classes import QueryAgentCollectionConfig
import logging
import os
import time
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Pure Weaviate Query Agent RAG API", version="3.0.0")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_KEY = os.getenv("WEAVIATE_API_KEY") or os.getenv("WEAVIATE_KEY")

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
    use_weaviate_intro_conclusion: Optional[bool] = True  # Use secondary Weaviate agent for intro/conclusion
    include_intro_conclusion: Optional[bool] = True
    include_debug_info: Optional[bool] = False  # Include raw agent response details

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
    intro_conclusion_method: Optional[str] = None  # "weaviate_agent" or "none"
    # Debug info (optional)
    raw_agent_info: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    return {
        "service": "Pure Weaviate Query Agent RAG API",
        "version": "3.0.0",
        "features": [
            "pure_weaviate_query_agents", 
            "detailed_source_metadata", 
            "optional_weaviate_intro_conclusion",
            "no_external_llm_dependencies"
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
        "agent_collections": ["MagisChunk"] if main_query_agent else [],
        "pure_weaviate": True,
        "external_llm_dependencies": False
    }

async def get_detailed_source_content(source_ids: List[str]) -> List[MagisChunkResult]:
    """
    Retrieve detailed content for source IDs from Weaviate
    This was the MISSING FUNCTION causing the 500 error
    """
    detailed_sources = []
    
    if not weaviate_client or not source_ids:
        logger.warning(f"No client or source IDs provided")
        return detailed_sources
    
    try:
        # Get the MagisChunk collection
        magis_chunk_collection = weaviate_client.collections.get("MagisChunk")
        
        for source_id in source_ids:
            try:
                # Fetch object by ID
                obj = magis_chunk_collection.query.fetch_object_by_id(
                    source_id,
                    return_properties=[
                        "content", "sourceFile", "headerContext", "humanId",
                        "agent1", "agent1CoreTopics", "citations", 
                        "chunkLength", "citationCount"
                    ]
                )
                
                if obj and obj.properties:
                    # Create MagisChunkResult from retrieved object
                    chunk_result = MagisChunkResult(
                        content=obj.properties.get("content", ""),
                        sourceFile=obj.properties.get("sourceFile", ""),
                        headerContext=obj.properties.get("headerContext", ""),
                        humanId=obj.properties.get("humanId", ""),
                        agent1=obj.properties.get("agent1", "Unknown"),
                        agent1CoreTopics=obj.properties.get("agent1CoreTopics", []),
                        citations=obj.properties.get("citations", []),
                        chunkLength=obj.properties.get("chunkLength", 0),
                        citationCount=obj.properties.get("citationCount", 0),
                        object_id=str(source_id),
                        distance=None
                    )
                    detailed_sources.append(chunk_result)
                    logger.info(f"✅ Retrieved details for source: {source_id}")
                else:
                    logger.warning(f"⚠️ No data found for source ID: {source_id}")
                    
            except Exception as e:
                logger.error(f"❌ Error fetching source {source_id}: {e}")
                continue
        
        logger.info(f"📚 Retrieved {len(detailed_sources)} detailed sources")
        return detailed_sources
        
    except Exception as e:
        logger.error(f"❌ Error in get_detailed_source_content: {e}")
        return detailed_sources

async def generate_weaviate_intro_conclusion(question: str, core_answer: str) -> Dict[str, str]:
    """Generate intro and conclusion using Weaviate agent"""
    
    if not intro_conclusion_agent:
        return {
            "intro": "Based on the theological and philosophical sources available:",
            "conclusion": "This information comes from our comprehensive knowledge base of theological and philosophical texts."
        }
    
    try:
        # Generate contextual intro using the agent
        intro_query = f"Provide a brief, welcoming introduction (1-2 sentences) for answering this question: {question}"
        intro_result = intro_conclusion_agent.run(intro_query)
        intro = str(intro_result.final_answer) if hasattr(intro_result, 'final_answer') else str(intro_result)
        
        # Generate contextual conclusion using the agent
        conclusion_query = f"Provide a brief, helpful conclusion (1-2 sentences) that wraps up this answer about '{question}'"
        conclusion_result = intro_conclusion_agent.run(conclusion_query)
        conclusion = str(conclusion_result.final_answer) if hasattr(conclusion_result, 'final_answer') else str(conclusion_result)
        
        # Keep them concise (truncate if too long)
        intro = intro[:300] + "..." if len(intro) > 300 else intro
        conclusion = conclusion[:300] + "..." if len(conclusion) > 300 else conclusion
        
        logger.info("✅ Generated intro/conclusion with Weaviate agent")
        return {"intro": intro, "conclusion": conclusion}
        
    except Exception as e:
        logger.error(f"Weaviate intro/conclusion generation error: {e}")
        return {
            "intro": "Based on the theological and philosophical sources available:",
            "conclusion": "This information comes from our comprehensive knowledge base of theological and philosophical texts."
        }

async def query_with_agent_plus_framing(question: str, use_weaviate_framing: bool = True, include_framing: bool = True, include_debug: bool = False) -> Dict[str, Any]:
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
                elif hasattr(source, 'uuid'):
                    source_ids.append(str(source.uuid))
        
        logger.info(f"🔗 Found {len(source_ids)} source IDs")
        
        # Get detailed source content - THIS WAS MISSING!
        detailed_sources = await get_detailed_source_content(source_ids)
        
        # Generate intro and conclusion if requested
        intro = None
        conclusion = None
        framing_method = None
        
        if include_framing and use_weaviate_framing:
            logger.info("🎭 Generating intro/conclusion with Weaviate agent...")
            framing = await generate_weaviate_intro_conclusion(question, core_answer)
            framing_method = "weaviate_agent"
            intro = framing.get("intro")
            conclusion = framing.get("conclusion")
        elif include_framing:
            # Simple default framing if Weaviate agent not available
            intro = "Based on the available theological and philosophical sources:"
            conclusion = "I hope this information from our knowledge base is helpful to you."
            framing_method = "simple_default"
        
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
        # Try to provide a more helpful error response
        if "503" in str(e):
            error_msg = "The query service is temporarily unavailable. Please try again later."
        elif "QueryAgent" in str(e):
            error_msg = "There was an issue with the query agent. The service may need to be restarted."
        else:
            error_msg = f"An error occurred while processing your query: {str(e)}"
        
        # Return a valid response instead of raising another exception
        return QueryResponse(
            intro=None,
            answer=error_msg,
            conclusion=None,
            magis_chunk_results=[],
            processing_time=time.time() - start_time,
            method_used="error_response",
            agent_confidence=0.0,
            intro_conclusion_method=None,
            raw_agent_info={"error": str(e)} if request.include_debug_info else None
        )

@app.post("/ask-pure-agent")
async def ask_pure_agent(request: QueryRequest):
    """Endpoint that uses only Weaviate Query Agent (no intro/conclusion)"""
    
    if not main_query_agent:
        raise HTTPException(status_code=503, detail="Query Agent not available")
    
    # Force no framing
    request.include_intro_conclusion = False
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
