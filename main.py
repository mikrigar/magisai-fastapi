#!/usr/bin/env python3
"""
Enhanced MagisAI FastAPI - Multi-Agent Processing
Handles both simple and complex questions with intelligent agent routing
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import weaviate
from weaviate.classes.init import Auth
import httpx
import asyncio
import logging
import os
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
import time
from collections import defaultdict

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Enhanced MagisAI Gateway", version="2.1.0")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_KEY = os.getenv("WEAVIATE_KEY")
RUNPOD_URL = os.getenv("RUNPOD_URL")
RUNPOD_KEY = os.getenv("RUNPOD_KEY")

# Validate required environment variables
required_vars = ["WEAVIATE_URL", "WEAVIATE_KEY", "RUNPOD_URL", "RUNPOD_KEY"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    logger.warning(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")

# Initialize Weaviate client
weaviate_client = None
try:
    if WEAVIATE_URL and WEAVIATE_KEY:
        weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_KEY)
        )
        logger.info("✅ Connected to Weaviate")
except Exception as e:
    logger.error(f"❌ Weaviate connection failed: {e}")


# Pydantic Models
class SubQuestion(BaseModel):
    question: str
    suggestedAgents: List[str] = Field(..., min_items=1)
    primaryAgent: Optional[str] = None


class Decomposition(BaseModel):
    isComplex: bool
    subQuestions: List[SubQuestion] = Field(..., min_items=1)
    synthesisStrategy: Optional[str] = Field(default="comprehensive")


class EnhancedQueryRequest(BaseModel):
    question: str
    user: str
    decomposition: Optional[Decomposition] = None
    maxChunks: Optional[int] = Field(default=5, ge=1, le=20)
    responseDepth: Optional[str] = Field(default="detailed")


class AgentResponse(BaseModel):
    agent: str
    subQuestion: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float = Field(..., ge=0.0, le=1.0)
    processingTime: float


class EnhancedQueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    agentResponses: List[AgentResponse]
    confidence: float = Field(..., ge=0.0, le=1.0)
    processingTime: float
    synthesisStrategy: str
    agentsUsed: List[str]


class SimpleQueryRequest(BaseModel):
    question: str
    user: str


# Constants
DEFAULT_PERSONA = (
    "You are a theological assistant to Fr. Robert Spitzer. "
    "Your job is to synthesize his work to answer questions clearly, "
    "confidently, and concisely. Do not mention your role or refer to yourself. "
    "Just answer the question based on Fr. Spitzer's material."
)
MAX_SOURCES_PER_AGENT = 3
MIN_CONFIDENCE_THRESHOLD = 0.5


@app.get("/")
async def root():
    """Root endpoint to check if the service is running."""
    return {
        "message": "MagisAI Enhanced Gateway v2 is running",
        "version": "2.1.0",
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "weaviate_connected": weaviate_client is not None,
        "environment_configured": all(os.getenv(var) for var in required_vars)
    }


async def query_agent_with_context(
    sub_question: str,
    agent: str,
    max_chunks: int,
    timeout: float = 30.0
) -> AgentResponse:
    """Query a specific agent with a sub-question."""
    logger.info(f"🔍 Querying agent '{agent}' for: {sub_question[:50]}...")
    
    payload = {
        "question": sub_question,
        "agent": agent,
        "max_chunks": max_chunks,
        "persona": DEFAULT_PERSONA
    }
    
    headers = {
        "Authorization": f"Bearer {RUNPOD_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                RUNPOD_URL,
                json=payload,
                headers=headers
            )
        
        elapsed_time = time.time() - start_time
        response.raise_for_status()
        data = response.json()
        
        return AgentResponse(
            agent=agent,
            subQuestion=sub_question,
            answer=data.get("answer", "No answer provided."),
            sources=data.get("sources", []),
            confidence=data.get("confidence", 0.8),
            processingTime=elapsed_time
        )
        
    except httpx.TimeoutException:
        logger.error(f"⏱️  Timeout querying agent {agent}")
        return _create_error_response(agent, sub_question, "Request timed out")
    except httpx.HTTPStatusError as e:
        logger.error(f"❌ HTTP error from agent {agent}: {e.response.status_code}")
        return _create_error_response(agent, sub_question, f"HTTP {e.response.status_code}")
    except Exception as e:
        logger.error(f"❌ Error querying agent {agent}: {str(e)}")
        return _create_error_response(agent, sub_question, "Internal error")


def _create_error_response(agent: str, sub_question: str, error_msg: str) -> AgentResponse:
    """Create a standardized error response."""
    return AgentResponse(
        agent=agent,
        subQuestion=sub_question,
        answer=f"Error from agent: {error_msg}",
        sources=[],
        confidence=0.0,
        processingTime=0.0
    )


def synthesize_responses(
    agent_responses: List[AgentResponse],
    synthesis_strategy: str
) -> str:
    """Synthesize multiple agent responses into a final answer."""
    # Filter responses by confidence threshold
    valid_responses = [
        r for r in agent_responses 
        if r.confidence >= MIN_CONFIDENCE_THRESHOLD
    ]
    
    if not valid_responses:
        return "Unable to generate a confident answer from available agents."
    
    # Sort by confidence for better synthesis
    valid_responses.sort(key=lambda x: x.confidence, reverse=True)
    
    if synthesis_strategy == "best_agent":
        return valid_responses[0].answer
    elif synthesis_strategy == "weighted":
        # Weight answers by confidence
        total_confidence = sum(r.confidence for r in valid_responses)
        if total_confidence == 0:
            return valid_responses[0].answer
        
        # For now, just concatenate in order of confidence
        # In a real implementation, you might want more sophisticated merging
        return " ".join(r.answer.strip() for r in valid_responses)
    else:  # comprehensive or default
        return " ".join(r.answer.strip() for r in valid_responses)


@app.post("/ask", response_model=EnhancedQueryResponse)
async def ask(request: EnhancedQueryRequest):
    """Process a complex multi-agent query."""
    if not request.decomposition:
        raise HTTPException(
            status_code=400,
            detail="Missing question decomposition. Please provide a decomposition object."
        )
    
    if not RUNPOD_URL or not RUNPOD_KEY:
        raise HTTPException(
            status_code=503,
            detail="Service not properly configured. Missing API credentials."
        )
    
    logger.info(f"🤖 Starting multi-agent query for user: {request.user}")
    logger.info(f"📋 Processing {len(request.decomposition.subQuestions)} sub-questions")
    
    # Create tasks for all agent queries
    agent_tasks = []
    for sub_q in request.decomposition.subQuestions:
        # Use primary agent if specified, otherwise use all suggested agents
        agents_to_query = [sub_q.primaryAgent] if sub_q.primaryAgent else sub_q.suggestedAgents
        
        for agent in agents_to_query:
            task = query_agent_with_context(
                sub_q.question,
                agent,
                request.maxChunks
            )
            agent_tasks.append(task)
    
    # Execute all queries in parallel
    start_time = time.time()
    agent_responses = await asyncio.gather(*agent_tasks)
    total_time = time.time() - start_time
    
    # Aggregate sources with capping
    source_aggregator = defaultdict(list)
    for response in agent_responses:
        if response.sources:
            source_aggregator[response.agent].extend(
                response.sources[:MAX_SOURCES_PER_AGENT]
            )
    
    # Flatten sources while maintaining diversity
    all_sources = []
    for agent_sources in source_aggregator.values():
        all_sources.extend(agent_sources)
    
    # Synthesize final answer
    synthesis_strategy = request.decomposition.synthesisStrategy or "comprehensive"
    final_answer = synthesize_responses(agent_responses, synthesis_strategy)
    
    # Calculate metrics
    agents_used = list(set(r.agent for r in agent_responses))
    avg_confidence = (
        sum(r.confidence for r in agent_responses) / len(agent_responses)
        if agent_responses else 0.0
    )
    
    logger.info(f"✅ Query completed in {total_time:.2f}s with {len(agents_used)} agents")
    
    return EnhancedQueryResponse(
        answer=final_answer,
        sources=all_sources,
        agentResponses=agent_responses,
        confidence=round(avg_confidence, 2),
        processingTime=round(total_time, 2),
        synthesisStrategy=synthesis_strategy,
        agentsUsed=agents_used
    )


@app.post("/ask-simple")
async def ask_simple(request: SimpleQueryRequest):
    """Process a simple single-agent query."""
    # This endpoint can be implemented for simple queries that don't need decomposition
    # For now, returning a placeholder
    return {
        "message": "Simple query endpoint not yet implemented",
        "question": request.question,
        "user": request.user
    }


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    if weaviate_client:
        try:
            weaviate_client.close()
            logger.info("✅ Weaviate connection closed")
        except Exception as e:
            logger.error(f"❌ Error closing Weaviate connection: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
