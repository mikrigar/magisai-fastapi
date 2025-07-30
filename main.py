#!/usr/bin/env python3
"""
Enhanced MagisAI FastAPI - Multi-Agent Processing
Handles both simple and complex questions with intelligent agent routing
"""
 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
 
load_dotenv()
 
app = FastAPI(title="Enhanced MagisAI Gateway", version="2.1.0")
 
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
 
# Models
class SubQuestion(BaseModel):
    question: str
    suggestedAgents: List[str]
    primaryAgent: Optional[str] = None
 
class Decomposition(BaseModel):
    isComplex: bool
    subQuestions: List[SubQuestion]
    synthesisStrategy: Optional[str] = "comprehensive"
 
class EnhancedQueryRequest(BaseModel):
    question: str
    user: str
    decomposition: Optional[Decomposition] = None
    maxChunks: Optional[int] = 5
    responseDepth: Optional[str] = "detailed"
 
class AgentResponse(BaseModel):
    agent: str
    subQuestion: str
    answer: str
    sources: List[Dict]
    confidence: float
    processingTime: float
 
class EnhancedQueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    agentResponses: List[AgentResponse]
    confidence: float
    processingTime: float
    synthesisStrategy: str
    agentsUsed: List[str]
 
class SimpleQueryRequest(BaseModel):
    question: str
    user: str

class SimpleQueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    confidence: float
    processingTime: float

@app.get("/")
async def root():
    return {"message": "MagisAI Enhanced Gateway v2 is running"}
 
# Helper to query agent with sub-question
async def query_agent_with_context(sub_question: str, agent: str, max_chunks: int) -> AgentResponse:
    logger.info(f"🔍 Querying agent '{agent}' for: {sub_question}")
    payload = {
        "question": sub_question,
        "agent": agent,
        "max_chunks": max_chunks,
        "persona": "You are a theological assistant to Fr. Robert Spitzer. Your job is to synthesize his work to answer questions clearly, confidently, and concisely. Do not mention your role or refer to yourself. Just answer the question based on Fr. Spitzer's material."
    }
    headers = {
        "Authorization": f"Bearer {RUNPOD_KEY}",
        "Content-Type": "application/json"
    }
    try:
        start = time.time()
        async with httpx.AsyncClient() as client:
            res = await client.post(f"{RUNPOD_URL}", json=payload, headers=headers, timeout=30.0)
        duration = time.time() - start
        res.raise_for_status()
        data = res.json()
 
        return AgentResponse(
            agent=agent,
            subQuestion=sub_question,
            answer=data.get("answer", "No answer provided."),
            sources=data.get("sources", []),
            confidence=data.get("confidence", 0.8),
            processingTime=duration
        )
    except Exception as e:
        logger.error(f"Error querying agent {agent}: {e}")
        return AgentResponse(
            agent=agent,
            subQuestion=sub_question,
            answer="Error from agent.",
            sources=[],
            confidence=0.0,
            processingTime=0.0
        )

# Simple query handler for non-decomposed questions
@app.post("/ask/simple", response_model=SimpleQueryResponse)
async def ask_simple(request: SimpleQueryRequest):
    """Handle simple, non-decomposed questions"""
    logger.info(f"📝 Simple query from {request.user}: {request.question}")
    
    # Default to a general agent for simple queries
    agent_response = await query_agent_with_context(
        request.question, 
        "general", 
        5  # default max chunks
    )
    
    return SimpleQueryResponse(
        answer=agent_response.answer,
        sources=agent_response.sources,
        confidence=agent_response.confidence,
        processingTime=agent_response.processingTime
    )
 
@app.post("/ask", response_model=EnhancedQueryResponse)
async def ask(request: EnhancedQueryRequest):
    if not request.decomposition:
        raise HTTPException(status_code=400, detail="Missing question decomposition")
 
    logger.info(f"🤖 Starting multi-agent query for user: {request.user}")
    agent_tasks = []
    for sub_q in request.decomposition.subQuestions:
        for agent in sub_q.suggestedAgents:
            agent_tasks.append(query_agent_with_context(sub_q.question, agent, request.maxChunks))
 
    raw_results = await asyncio.gather(*agent_tasks)
 
    # Smarter source capping
    grouped_sources = defaultdict(list)
    for r in raw_results:
        grouped_sources[r.agent].extend(r.sources[:3])
    flattened_sources = [s for group in grouped_sources.values() for s in group]
 
    # Synthesize response
    final_answer = " ".join(
        r.answer.strip() for r in raw_results if r.confidence > 0.5
    ).strip()
    
    agents_used = list(set(r.agent for r in raw_results))
    total_conf = sum(r.confidence for r in raw_results) / len(raw_results) if raw_results else 0.0
    total_time = sum(r.processingTime for r in raw_results)
 
    return EnhancedQueryResponse(
        answer=final_answer,
        sources=flattened_sources,
        agentResponses=raw_results,
        confidence=round(total_conf, 2),
        processingTime=round(total_time, 2),
        synthesisStrategy=request.decomposition.synthesisStrategy or "comprehensive",
        agentsUsed=agents_used
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check the health of the service and its dependencies"""
    health_status = {
        "status": "healthy",
        "weaviate": "connected" if weaviate_client else "disconnected",
        "runpod": "configured" if RUNPOD_URL and RUNPOD_KEY else "not configured"
    }
    
    if not weaviate_client or not RUNPOD_URL:
        health_status["status"] = "degraded"
    
    return health_status

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    if weaviate_client:
        try:
            weaviate_client.close()
            logger.info("✅ Weaviate client closed")
        except Exception as e:
            logger.error(f"Error closing Weaviate client: {e}")
