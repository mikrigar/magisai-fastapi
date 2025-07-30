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

load_dotenv()

app = FastAPI(title="Enhanced MagisAI Gateway", version="2.0.0")

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

# Enhanced request models
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

# Legacy request model for backward compatibility
class SimpleQueryRequest(BaseModel):
    question: str
    user: str

@app.get("/")
async def root():
    return {
        "service": "Enhanced MagisAI Gateway",
        "version": "2.0.0",
        "capabilities": ["multi-agent processing", "question decomposition", "intelligent synthesis"],
        "endpoints": {
            "ask": "/ask (POST) - Legacy single-agent endpoint",
            "ask-enhanced": "/ask-enhanced (POST) - Multi-agent endpoint",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "weaviate_connected": weaviate_client is not None and weaviate_client.is_ready(),
        "runpod_configured": bool(RUNPOD_URL and RUNPOD_KEY),
        "capabilities": ["multi-agent", "parallel-processing", "intelligent-synthesis"]
    }

async def query_agent_with_context(question: str, agent_filter: str, max_chunks: int = 5) -> Dict[str, Any]:
    """Query specific agent with Weaviate context retrieval"""
    
    if not weaviate_client:
        raise HTTPException(status_code=503, detail="Weaviate not connected")
    
    start_time = time.time()
    
    try:
        collection = weaviate_client.collections.get("MagisChunk")
        
        # Search with agent filtering
        response = collection.query.near_text(
            query=question,
            limit=max_chunks,
            return_metadata=["distance"],
            where={
                "operator": "Or",
                "operands": [
                    {"path": ["agent1"], "operator": "Equal", "valueString": agent_filter},
                    {"path": ["agent2"], "operator": "Equal", "valueString": agent_filter}
                ]
            }
        )
        
        # Format chunks
        chunks = []
        for obj in response.objects:
            chunks.append({
                "content": obj.properties.get("content", ""),
                "agent": obj.properties.get("agent1", "Unknown"),
                "source": obj.properties.get("sourceFile", "Unknown"),
                "distance": obj.metadata.distance if hasattr(obj.metadata, 'distance') else 1.0
            })
        
        # Generate answer with RunPod Mistral
        answer = await generate_agent_answer(question, chunks, agent_filter)
        
        # Calculate confidence
        avg_distance = sum(chunk["distance"] for chunk in chunks) / len(chunks) if chunks else 1.0
        confidence = max(0.0, 1.0 - avg_distance)
        
        processing_time = time.time() - start_time
        
        return {
            "agent": agent_filter,
            "question": question,
            "answer": answer,
            "sources": chunks,
            "confidence": confidence,
            "processing_time": processing_time,
            "chunks_used": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"❌ Agent query failed for {agent_filter}: {e}")
        return {
            "agent": agent_filter,
            "question": question,
            "answer": f"Unable to process question with {agent_filter.replace('_', ' ')} agent due to technical difficulties.",
            "sources": [],
            "confidence": 0.0,
            "processing_time": time.time() - start_time,
            "error": str(e)
        }

async def generate_agent_answer(question: str, chunks: List[Dict], agent: str) -> str:
    """Generate agent-specific answer using RunPod Mistral"""
    
    if not chunks:
        return f"I apologize, but I couldn't find relevant information from {agent.replace('_', ' ')} sources to answer your question."
    
    # Build context
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[Source {i}]: {chunk['content']}")
    
    context = "\n\n".join(context_parts)
    
    # Agent-specific prompt
    agent_name = agent.replace("_", " ")
    prompt = f"""You are a Catholic theology expert specializing in {agent_name}. Answer the user's question using the provided sources.

Sources from {agent_name}:
{context}

Question: {question}

Provide a clear, theologically sound answer based on these {agent_name} sources. If the sources don't fully address the question, acknowledge the limitations while providing what insight you can."""

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                RUNPOD_URL,
                json={
                    "model": "mistral-nemo",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 400,
                    "temperature": 0.7,
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
            
            raise Exception(f"RunPod error: {response.status_code}")
            
    except Exception as e:
        logger.error(f"❌ RunPod generation failed: {e}")
        return f"Based on {len(chunks)} sources from {agent_name}, this question relates to important theological concepts. However, I'm experiencing technical difficulties generating a detailed response. Please try again shortly."

async def synthesize_multi_agent_responses(agent_responses: List[Dict], original_question: str, strategy: str = "comprehensive") -> str:
    """Synthesize multiple agent responses into unified answer"""
    
    if not agent_responses:
        return "I apologize, but I couldn't generate a response to your question."
    
    if len(agent_responses) == 1:
        return agent_responses[0]["answer"]
    
    # Prepare synthesis context
    agent_answers = []
    for resp in agent_responses:
        agent_name = resp["agent"].replace("_", " ")
        agent_answers.append(f"**{agent_name} Perspective:**\n{resp['answer']}")
    
    combined_context = "\n\n".join(agent_answers)
    
    synthesis_prompt = f"""You are MagisAI, a Catholic theology assistant. Synthesize the following expert perspectives into one comprehensive, coherent answer.

Original Question: {original_question}

Expert Perspectives:
{combined_context}

Synthesize these perspectives into a unified, comprehensive answer that:
1. Addresses the original question directly
2. Integrates insights from all perspectives
3. Maintains theological accuracy
4. Presents a coherent Catholic understanding
5. Is concise but thorough

Unified Answer:"""

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                RUNPOD_URL,
                json={
                    "model": "mistral-nemo",
                    "messages": [{"role": "user", "content": synthesis_prompt}],
                    "max_tokens": 600,
                    "temperature": 0.6,
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
        
        # Fallback synthesis
        return f"Based on multiple theological perspectives including {', '.join([r['agent'].replace('_', ' ') for r in agent_responses])}, here are the key insights:\n\n" + "\n\n".join([resp["answer"] for resp in agent_responses[:2]])
        
    except Exception as e:
        logger.error(f"❌ Synthesis failed: {e}")
        # Simple fallback
        return agent_responses[0]["answer"] + f"\n\n(Additional perspectives from {len(agent_responses)-1} other theological domains were considered in this response.)"

@app.post("/ask-enhanced", response_model=EnhancedQueryResponse)
async def enhanced_query(request: EnhancedQueryRequest):
    """Enhanced multi-agent query processing"""
    
    start_time = time.time()
    logger.info(f"🔍 Enhanced query from {request.user}: {request.question}")
    
    try:
        # If no decomposition provided, treat as simple question
        if not request.decomposition or not request.decomposition.isComplex:
            logger.info("📝 Processing as simple question")
            
            # Default to Christian Theology for simple questions
            agent_response = await query_agent_with_context(
                request.question, 
                "Christian_Theology", 
                request.maxChunks
            )
            
            return EnhancedQueryResponse(
                answer=agent_response["answer"],
                sources=agent_response["sources"],
                agentResponses=[AgentResponse(
                    agent=agent_response["agent"],
                    subQuestion=request.question,
                    answer=agent_response["answer"],
                    sources=agent_response["sources"],
                    confidence=agent_response["confidence"],
                    processingTime=agent_response["processing_time"]
                )],
                confidence=agent_response["confidence"],
                processingTime=time.time() - start_time,
                synthesisStrategy="single_agent",
                agentsUsed=[agent_response["agent"]]
            )
        
        # Complex question - multi-agent processing
        logger.info(f"🔄 Processing complex question with {len(request.decomposition.subQuestions)} sub-questions")
        
        # Collect all unique agents
        all_agents = set()
        for sub_q in request.decomposition.subQuestions:
            all_agents.update(sub_q.suggestedAgents)
        
        # Query each agent in parallel
        agent_tasks = []
        for agent in all_agents:
            # Find the most relevant sub-question for this agent
            relevant_question = request.question  # Default to main question
            for sub_q in request.decomposition.subQuestions:
                if agent in sub_q.suggestedAgents:
                    relevant_question = sub_q.question
                    break
            
            task = query_agent_with_context(relevant_question, agent, request.maxChunks)
            agent_tasks.append(task)
        
        # Execute all agent queries in parallel
        agent_responses = await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        # Filter successful responses
        successful_responses = [
            resp for resp in agent_responses 
            if isinstance(resp, dict) and not resp.get("error")
        ]
        
        if not successful_responses:
            raise HTTPException(status_code=500, detail="All agent queries failed")
        
        # Synthesize responses
        synthesized_answer = await synthesize_multi_agent_responses(
            successful_responses, 
            request.question, 
            request.decomposition.synthesisStrategy
        )
        
        # Combine all sources
        all_sources = []
        for resp in successful_responses:
            all_sources.extend(resp["sources"])
        
        # Calculate overall confidence
        avg_confidence = sum(resp["confidence"] for resp in successful_responses) / len(successful_responses)
        
        # Create agent response objects
        agent_response_objects = [
            AgentResponse(
                agent=resp["agent"],
                subQuestion=resp["question"],
                answer=resp["answer"],
                sources=resp["sources"],
                confidence=resp["confidence"],
                processingTime=resp["processing_time"]
            )
            for resp in successful_responses
        ]
        
        return EnhancedQueryResponse(
            answer=synthesized_answer,
            sources=all_sources[:10],  # Limit to top 10 sources
            agentResponses=agent_response_objects,
            confidence=avg_confidence,
            processingTime=time.time() - start_time,
            synthesisStrategy=request.decomposition.synthesisStrategy,
            agentsUsed=list(all_agents)
        )
        
    except Exception as e:
        logger.error(f"❌ Enhanced query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Keep legacy endpoint for backward compatibility
@app.post("/ask")
async def simple_query(request: SimpleQueryRequest):
    """Legacy single-agent endpoint"""
    
    enhanced_request = EnhancedQueryRequest(
        question=request.question,
        user=request.user
    )
    
    response = await enhanced_query(enhanced_request)
    
    # Return simplified response for backward compatibility
    return {
        "answer": response.answer,
        "sources": response.sources
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
