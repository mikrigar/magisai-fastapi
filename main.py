#!/usr/bin/env python3
"""
Enhanced FastAPI RAG Service with Weaviate Query Agent
Uses Weaviate's intelligent query agent for better results
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

app = FastAPI(title="Enhanced RAG API with Query Agent", version="2.0.0")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_KEY = os.getenv("WEAVIATE_API_KEY") or os.getenv("WEAVIATE_KEY")
RUNPOD_URL = os.getenv("RUNPOD_URL")
RUNPOD_KEY = os.getenv("RUNPOD_KEY")

# Initialize Weaviate client and Query Agent
weaviate_client = None
query_agent = None

try:
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_KEY)
    )
    
    # Initialize the Query Agent
    query_agent = QueryAgent(
        client=weaviate_client,
        collections=[
            QueryAgentCollectionConfig(
                name="MagisChunk",
                target_vector=["content_vector"],  # Adjust if your vector name is different
            ),
        ],
    )
    
    logger.info("✅ Connected to Weaviate with Query Agent")
except Exception as e:
    logger.error(f"❌ Weaviate/Query Agent setup failed: {e}")

class QueryRequest(BaseModel):
    question: str
    user: str
    force_method: Optional[str] = None  # "agent", "basic", or None for auto-detect
    max_chunks: Optional[int] = 5

class AgentSource(BaseModel):
    content: str
    source: str
    metadata: Dict[str, Any] = {}

class QueryResponse(BaseModel):
    answer: str
    sources: List[AgentSource]
    processing_time: float
    method_used: str  # "query_agent" or "basic_search"
    complexity_assessment: str  # "simple", "moderate", "complex"
    reasoning: str  # Why this method was chosen
    agent_confidence: Optional[float] = None

@app.get("/")
async def root():
    return {
        "service": "Enhanced RAG API with Query Agent",
        "version": "2.0.0",
        "features": ["weaviate_query_agent", "intelligent_retrieval", "enhanced_reasoning"],
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "weaviate_connected": weaviate_client is not None and weaviate_client.is_ready(),
        "query_agent_ready": query_agent is not None,
        "runpod_configured": bool(RUNPOD_URL and RUNPOD_KEY),
        "intelligent_routing": True,
        "agent_collections": ["MagisChunk"] if query_agent else []
    }

async def assess_query_complexity(question: str) -> Dict[str, str]:
    """Use Mistral to assess query complexity and recommend processing method"""
    
    if not RUNPOD_URL or not RUNPOD_KEY:
        # Fallback to simple heuristics if no LLM available
        return simple_complexity_heuristic(question)
    
    complexity_prompt = f"""Analyze this question and determine its complexity for a theological/philosophical knowledge system:

Question: "{question}"

Classify the complexity and recommend the best processing method:

COMPLEXITY LEVELS:
- SIMPLE: Basic factual questions, definitions, yes/no questions, single concept queries
- MODERATE: Questions requiring some reasoning but straightforward to answer
- COMPLEX: Multi-faceted questions, requires deep reasoning, philosophical analysis, connecting multiple concepts, comparative analysis

PROCESSING METHODS:
- basic_search: For simple/moderate questions that can be answered with direct retrieval + basic generation
- query_agent: For complex questions requiring intelligent reasoning, context understanding, and sophisticated retrieval

Respond in exactly this format:
COMPLEXITY: [simple/moderate/complex]
METHOD: [basic_search/query_agent]
REASONING: [One sentence explanation of why this method is best]"""

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                RUNPOD_URL,
                json={
                    "model": "mistralai/Mistral-Small-Instruct-2409",
                    "messages": [{"role": "user", "content": complexity_prompt}],
                    "max_tokens": 150,
                    "temperature": 0.1,  # Low temperature for consistent analysis
                    "stream": False
                },
                headers={"Authorization": f"Bearer {RUNPOD_KEY}"},
                timeout=15.0
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("choices") and result["choices"][0].get("message"):
                    analysis = result["choices"][0]["message"]["content"].strip()
                    return parse_complexity_response(analysis)
                elif result.get("response"):
                    analysis = result["response"].strip()
                    return parse_complexity_response(analysis)
            
            logger.warning("LLM complexity assessment failed, using heuristics")
            return simple_complexity_heuristic(question)
            
    except Exception as e:
        logger.error(f"Complexity assessment error: {e}")
        return simple_complexity_heuristic(question)

def parse_complexity_response(analysis: str) -> Dict[str, str]:
    """Parse the LLM response for complexity assessment"""
    
    complexity = "moderate"
    method = "basic_search"
    reasoning = "Default fallback decision"
    
    try:
        lines = analysis.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("COMPLEXITY:"):
                complexity = line.split(":", 1)[1].strip().lower()
            elif line.startswith("METHOD:"):
                method = line.split(":", 1)[1].strip().lower()
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
    except Exception as e:
        logger.error(f"Failed to parse complexity response: {e}")
    
    return {
        "complexity": complexity,
        "method": method,
        "reasoning": reasoning
    }

def simple_complexity_heuristic(question: str) -> Dict[str, str]:
    """Fallback heuristic-based complexity assessment"""
    
    question_lower = question.lower()
    
    # Simple question indicators
    simple_indicators = [
        "what is", "who is", "when did", "where is", "define", "meaning of",
        "yes or no", "true or false", "is it", "does it", "can you"
    ]
    
    # Complex question indicators  
    complex_indicators = [
        "why does", "how does", "explain the relationship", "compare", "analyze",
        "what are the implications", "philosophical", "theological significance",
        "consciousness", "quantum", "existence", "metaphysical", "epistemological",
        "prove that", "demonstrate", "argue for", "justify"
    ]
    
    if any(indicator in question_lower for indicator in complex_indicators):
        return {
            "complexity": "complex",
            "method": "query_agent", 
            "reasoning": "Question contains complex philosophical/theological concepts requiring intelligent analysis"
        }
    elif any(indicator in question_lower for indicator in simple_indicators):
        return {
            "complexity": "simple",
            "method": "basic_search",
            "reasoning": "Simple factual question suitable for direct retrieval"
        }
    else:
        return {
            "complexity": "moderate", 
            "method": "basic_search",
            "reasoning": "Moderate complexity question suitable for enhanced retrieval"
        }

async def query_with_agent(question: str) -> Dict[str, Any]:
    """Use Weaviate Query Agent for intelligent retrieval and reasoning"""
    
    if not query_agent:
        raise HTTPException(status_code=503, detail="Query Agent not available")
    
    try:
        # Run the query agent
        result = query_agent.run(question)
        
        # Extract the answer and sources from agent result
        # The agent typically returns a structured response
        answer = str(result.answer) if hasattr(result, 'answer') else str(result)
        
        # Try to extract sources if available
        sources = []
        if hasattr(result, 'sources') and result.sources:
            for source in result.sources:
                agent_source = AgentSource(
                    content=getattr(source, 'content', str(source)),
                    source=getattr(source, 'source', 'Query Agent Result'),
                    metadata=getattr(source, 'metadata', {})
                )
                sources.append(agent_source)
        else:
            # If no explicit sources, create one from the result
            sources.append(AgentSource(
                content=answer,
                source="Weaviate Query Agent",
                metadata={"method": "intelligent_query_agent"}
            ))
        
        logger.info(f"✅ Query Agent processed successfully")
        return {
            "answer": answer,
            "sources": sources,
            "confidence": getattr(result, 'confidence', 0.9)
        }
        
    except Exception as e:
        logger.error(f"❌ Query Agent failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query Agent error: {str(e)}")

async def basic_search_with_generation(question: str, max_chunks: int = 5) -> Dict[str, Any]:
    """Basic vector search + Mistral generation for simple/moderate questions"""
    
    if not weaviate_client:
        raise HTTPException(status_code=503, detail="Weaviate not connected")
    
    try:
        # Get relevant chunks
        collection = weaviate_client.collections.get("MagisChunk")
        
        response = collection.query.near_text(
            query=question,
            limit=max_chunks,
            return_metadata=["distance"],
            return_properties=["content", "sourceFile"]
        )
        
        sources = []
        context_parts = []
        
        for i, obj in enumerate(response.objects, 1):
            content = obj.properties.get("content", "")
            source_file = obj.properties.get("sourceFile", "Unknown")
            
            source = AgentSource(
                content=content,
                source=source_file,
                metadata={
                    "distance": obj.metadata.distance if hasattr(obj.metadata, 'distance') else 1.0,
                    "method": "basic_search_with_generation"
                }
            )
            sources.append(source)
            context_parts.append(f"[{i}] {content}")
        
        if not sources:
            return {
                "answer": "No relevant information found to answer your question.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Generate answer using Mistral
        context = "\n\n".join(context_parts)
        
        generation_prompt = f"""Question: {question}

Context from theological sources:
{context}

Please provide a clear, accurate answer based on the context above. Use the information provided to give a helpful response."""

        # Generate with Mistral
        async with httpx.AsyncClient() as client:
            response = await client.post(
                RUNPOD_URL,
                json={
                    "model": "mistralai/Mistral-Small-Instruct-2409",
                    "messages": [{"role": "user", "content": generation_prompt}],
                    "max_tokens": 800,
                    "temperature": 0.3,
                    "stream": False
                },
                headers={"Authorization": f"Bearer {RUNPOD_KEY}"},
                timeout=30.0
            )
            
            answer = "Based on the retrieved sources, I can provide relevant information, but please try again for a complete response."
            
            if response.status_code == 200:
                result = response.json()
                if result.get("choices") and result["choices"][0].get("message"):
                    answer = result["choices"][0]["message"]["content"].strip()
                elif result.get("response"):
                    answer = result["response"].strip()
        
        logger.info(f"✅ Basic search + generation completed with {len(sources)} sources")
        return {
            "answer": answer,
            "sources": sources,
            "confidence": 0.8 if sources else 0.0
        }
        
    except Exception as e:
        logger.error(f"❌ Basic search + generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search and generation error: {str(e)}")

async def basic_vector_search(question: str, max_chunks: int = 5) -> Dict[str, Any]:
    """Fallback to basic vector search"""
    
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
            source = AgentSource(
                content=obj.properties.get("content", ""),
                source=obj.properties.get("sourceFile", "Unknown"),
                metadata={
                    "distance": obj.metadata.distance if hasattr(obj.metadata, 'distance') else 1.0,
                    "method": "basic_vector_search"
                }
            )
            sources.append(source)
        
        # Create basic answer from first source
        answer = f"Based on the retrieved information: {sources[0].content[:500]}..." if sources else "No relevant information found."
        
        logger.info(f"✅ Basic search retrieved {len(sources)} sources")
        return {
            "answer": answer,
            "sources": sources,
            "confidence": 0.7 if sources else 0.0
        }
        
    except Exception as e:
        logger.error(f"❌ Basic search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Main query endpoint with intelligent complexity-based routing"""
    
    start_time = time.time()
    logger.info(f"Query from {request.user}: {request.question}")
    
    try:
        # Assess complexity and choose method (unless forced)
        if request.force_method:
            if request.force_method == "agent":
                assessment = {"complexity": "complex", "method": "query_agent", "reasoning": "Forced to use query agent"}
            else:
                assessment = {"complexity": "simple", "method": "basic_search", "reasoning": "Forced to use basic search"}
        else:
            assessment = await assess_query_complexity(request.question)
        
        logger.info(f"Complexity assessment: {assessment['complexity']} -> {assessment['method']}")
        
        # Route to appropriate method
        if assessment["method"] == "query_agent" and query_agent:
            result = await query_with_agent(request.question)
            method_used = "query_agent"
        else:
            # Use basic search + Mistral generation for simple/moderate questions
            result = await basic_search_with_generation(request.question, request.max_chunks)
            method_used = "basic_search_with_generation"
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            processing_time=time.time() - start_time,
            method_used=method_used,
            complexity_assessment=assessment["complexity"],
            reasoning=assessment["reasoning"],
            agent_confidence=result.get("confidence")
        )
        
    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-agent")
async def ask_with_agent_only(request: QueryRequest):
    """Endpoint that forces use of Query Agent only"""
    
    if not query_agent:
        raise HTTPException(status_code=503, detail="Query Agent not available")
    
    # Force agent usage
    request.force_method = "agent"
    return await ask_question(request)

@app.post("/ask-basic")
async def ask_with_basic_search_only(request: QueryRequest):
    """Endpoint that forces use of basic search + generation only"""
    
    # Force basic search
    request.force_method = "basic"
    return await ask_question(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
