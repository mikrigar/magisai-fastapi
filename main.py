#!/usr/bin/env python3
"""
Simplified MagisAI RAG - Focus on Quality Citations and Proofs
Optimized for theological accuracy with enhanced attribution
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import weaviate
from weaviate.classes.init import Auth
import httpx
import logging
import os
import re
import time
from typing import List, Dict, Optional
from dotenv import load_dotenv


load_dotenv()

app = FastAPI(title="Simplified MagisAI RAG", version="3.0.0")

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

class SimplifiedQueryRequest(BaseModel):
    question: str
    user: str
    maxChunks: Optional[int] = 8
    includeProofs: Optional[bool] = True
    citationStyle: Optional[str] = "detailed"  # "detailed" or "inline"

class EnhancedSource(BaseModel):
    content: str
    agent: str
    source: str
    humanId: str
    distance: float
    hasDetailedContent: bool = False
    namedPersons: List[str] = []
    contentType: Optional[str] = None
    hasEnumeration: bool = False
    detailLevel: int = 0

class SimplifiedResponse(BaseModel):
    answer: str
    sources: List[EnhancedSource]
    confidence: float
    processingTime: float
    citationsUsed: int
    detailedContentUsed: int
    breakdownsIncluded: int

@app.get("/")
async def root():
    return {
        "service": "Simplified MagisAI RAG",
        "version": "3.0.0",
        "focus": ["theological accuracy", "enhanced citations", "proof attribution"],
        "ai_model": "Mistral-Small 3.1 24B with 128k context",
        "vector_db": "Weaviate with Fr. Spitzer's complete works"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "weaviate_connected": weaviate_client is not None and weaviate_client.is_ready(),
        "runpod_configured": bool(RUNPOD_URL and RUNPOD_KEY),
        "focus": "simplified_rag_with_enhanced_citations"
    }

def extract_persons_and_detailed_content(content: str) -> Dict[str, any]:
    """Extract named persons and identify detailed breakdowns, explanations, and proofs"""
    
    # Common theological/philosophical names to look for
    theological_names = [
        "Aquinas", "Augustine", "Aristotle", "Plato", "Descartes", "Kant", 
        "Anselm", "Bonaventure", "Scotus", "Ockham", "Newman", "Lonergan",
        "Rahner", "Balthasar", "Ratzinger", "Benedict", "Francis", "John Paul",
        "Paul VI", "Pius", "Leo XIII", "Gregory", "Chrysostom", "Jerome",
        "Ambrose", "Basil", "Athanasius", "Ignatius", "Teresa", "John of the Cross",
        "Spitzer", "Einstein", "Hawking", "Heisenberg", "Planck", "Darwin",
        "Mendel", "Teilhard", "Lemaitre", "Pascal", "Newton", "Copernicus"
    ]
    
    # Expanded patterns for detailed content (proofs, breakdowns, explanations)
    detailed_content_patterns = [
        # Formal proofs
        r"proof\s+(?:of|for|that)",
        r"theorem\s+(?:of|states)",
        r"Q\.E\.D\.",
        r"demonstration\s+(?:of|that)",
        
        # Conceptual breakdowns and explanations
        r"breakdown\s+(?:of|into)",
        r"explanation\s+(?:of|for)",
        r"analysis\s+(?:of|shows)",
        r"definition\s+(?:of|states)",
        r"consists?\s+of",
        r"composed\s+of",
        r"divided\s+into",
        r"categorized\s+(?:into|as)",
        r"classified\s+(?:into|as)",
        r"structured\s+(?:into|as)",
        
        # Step-by-step explanations
        r"first(?:ly)?[,\s]",
        r"second(?:ly)?[,\s]",
        r"third(?:ly)?[,\s]",
        r"next[,\s]",
        r"then[,\s]",
        r"finally[,\s]",
        r"in\s+conclusion[,\s]",
        r"step\s+(?:one|two|three|\d+)",
        r"stage\s+(?:one|two|three|\d+)",
        r"phase\s+(?:one|two|three|\d+)",
        
        # Detailed descriptions
        r"detailed\s+(?:explanation|description|analysis)",
        r"comprehensive\s+(?:explanation|description|analysis)",
        r"thorough\s+(?:explanation|description|analysis)",
        r"in\s+detail[,\s]",
        r"specifically[,\s]",
        r"more\s+precisely[,\s]",
        
        # Logical reasoning
        r"therefore[,\s]",
        r"thus[,\s]",
        r"hence[,\s]",
        r"consequently[,\s]",
        r"as\s+a\s+result[,\s]",
        r"it\s+follows\s+that",
        r"we\s+can\s+conclude",
        r"this\s+(?:means|implies|suggests)",
        
        # Enumerated lists and structures
        r"(?:a\)|1\.|i\.|•|\*)\s+",
        r"(?:b\)|2\.|ii\.)",
        r"(?:c\)|3\.|iii\.)",
        r"namely[,:\s]",
        r"such\s+as[,:\s]",
        r"for\s+example[,:\s]",
        r"for\s+instance[,:\s]",
        
        # Scientific/mathematical details
        r"equation\s+(?:shows|demonstrates|states)",
        r"formula\s+(?:proves|shows|demonstrates)",
        r"calculation\s+(?:shows|demonstrates)",
        r"experiment\s+(?:shows|demonstrates|proves)",
        r"study\s+(?:shows|demonstrates|found)",
        r"research\s+(?:shows|demonstrates|indicates)",
        
        # Theological/philosophical structures
        r"doctrine\s+(?:teaches|states|holds)",
        r"tradition\s+(?:teaches|states|holds)",
        r"church\s+teaches",
        r"scripture\s+(?:teaches|states|says)",
        r"revelation\s+(?:teaches|shows)",
        r"magisterium\s+(?:teaches|states)"
    ]
    
    content_lower = content.lower()
    
    # Find persons mentioned
    persons_found = []
    for name in theological_names:
        if name.lower() in content_lower:
            persons_found.append(name)
    
    # Check for detailed content (proofs, breakdowns, explanations)
    has_detailed_content = False
    content_type = None
    detail_indicators = []
    
    for pattern in detailed_content_patterns:
        if re.search(pattern, content_lower):
            has_detailed_content = True
            detail_indicators.append(pattern)
    
    # Determine content type based on indicators found
    if has_detailed_content:
        if any(re.search(p, content_lower) for p in ["proof", "theorem", "q\\.e\\.d\\.", "demonstration"]):
            content_type = "formal_proof"
        elif any(re.search(p, content_lower) for p in ["breakdown", "divided", "categorized", "classified", "structured"]):
            content_type = "conceptual_breakdown"
        elif any(re.search(p, content_lower) for p in ["first", "second", "step", "stage", "phase"]):
            content_type = "step_by_step_explanation"
        elif any(re.search(p, content_lower) for p in ["equation", "formula", "calculation", "experiment"]):
            content_type = "scientific_mathematical"
        elif any(re.search(p, content_lower) for p in ["doctrine", "tradition", "scripture", "revelation"]):
            content_type = "theological_explanation"
        else:
            content_type = "detailed_explanation"
    
    # Additional check for numbered/bulleted lists
    has_enumeration = bool(re.search(r"(?:1\.|2\.|3\.|a\)|b\)|c\)|•|\*|\-)\s+", content))
    
    return {
        "persons": persons_found,
        "has_detailed_content": has_detailed_content,
        "content_type": content_type,
        "has_enumeration": has_enumeration,
        "detail_count": len(detail_indicators)
    }

async def enhanced_weaviate_search(question: str, max_chunks: int = 8) -> List[EnhancedSource]:
    """Enhanced Weaviate search with proof and person extraction"""
    
    if not weaviate_client:
        raise HTTPException(status_code=503, detail="Weaviate not connected")
    
    try:
        collection = weaviate_client.collections.get("MagisChunk")
        
        # Enhanced search with better metadata
        response = collection.query.near_text(
            query=question,
            limit=max_chunks,
            return_metadata=["distance", "score"],
            return_properties=["content", "agent1", "agent1CoreTopics", "sourceFile", "humanId"]
        )
        
        enhanced_sources = []
        
        for obj in response.objects:
            content = obj.properties.get("content", "")
            
            # Extract persons and detailed content from content
            analysis = extract_persons_and_detailed_content(content)
            
            enhanced_source = EnhancedSource(
                content=content,
                agent=obj.properties.get("agent1", "Unknown"),
                source=obj.properties.get("sourceFile", "Unknown"),
                humanId=obj.properties.get("humanId", ""),
                distance=obj.metadata.distance if hasattr(obj.metadata, 'distance') else 1.0,
                hasDetailedContent=analysis["has_detailed_content"],
                namedPersons=analysis["persons"],
                contentType=analysis["content_type"],
                hasEnumeration=analysis["has_enumeration"],
                detailLevel=analysis["detail_count"]
            )
            
            enhanced_sources.append(enhanced_source)
        
        logger.info(f"✅ Retrieved {len(enhanced_sources)} enhanced chunks")
        return enhanced_sources
        
    except Exception as e:
        logger.error(f"❌ Enhanced search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

async def generate_enhanced_answer(question: str, sources: List[EnhancedSource], include_proofs: bool = True) -> str:
    """Generate answer with enhanced citations using Mistral-Small 3.1"""
    
    if not sources:
        return "I apologize, but I couldn't find relevant information to answer your question."
    
    # Build enhanced context with proof and citation information
    context_parts = []
    citation_map = {}
    
    for i, source in enumerate(sources, 1):
        citation_key = f"[{i}]"
        citation_map[citation_key] = {
            "source": source.source,
            "agent": source.agent,
            "persons": source.namedPersons,
            "proof_type": source.proofType
        }
        
        # Enhanced context formatting
        context_part = f"{citation_key} From {source.agent.replace('_', ' ')} ({source.source}):\n{source.content}"
        
        if source.namedPersons:
            context_part += f"\n[Notable persons mentioned: {', '.join(source.namedPersons)}]"
        
        if source.hasDetailedContent and source.contentType:
            context_part += f"\n[Contains {source.contentType.replace('_', ' ')} - Detail Level: {source.detailLevel}]"
            
        if source.hasEnumeration:
            context_part += f"\n[Contains numbered/structured breakdown]"
        
        context_parts.append(context_part)
    
    context = "\n\n".join(context_parts)
    
    # Enhanced prompt for Mistral-Small 3.1 with citation requirements
    citation_instructions = """
CRITICAL CITATION AND DETAIL REQUIREMENTS:
1. When sources contain detailed breakdowns, explanations, or conceptual analyses, include those complete details in your answer
2. When sources contain step-by-step explanations, preserve the step-by-step structure
3. When sources contain numbered lists, categorizations, or structured breakdowns, incorporate that structure
4. When referencing mathematical, scientific, philosophical, or theological proofs, include detailed explanations from the sources
5. When mentioning named persons, always attribute their contributions properly and include their specific insights
6. Use [1], [2], [3] etc. to cite sources throughout your answer
7. If a source contains enumerated points or detailed breakdowns, include those specifics rather than general summaries
8. Preserve the logical structure and organization present in the source material
9. Include definitions, categorizations, and conceptual frameworks when present in sources
10. Maintain theological accuracy while being comprehensive and detailed
"""

    prompt = f"""You are MagisAI, a Catholic theology assistant powered by Fr. Spitzer's authoritative works. Answer the user's question with enhanced citations and detailed attribution.

{citation_instructions}

QUESTION: {question}

AUTHORITATIVE SOURCES:
{context}

INSTRUCTIONS:
- Provide a comprehensive, theologically sound answer
- Include ALL detailed breakdowns, explanations, and structured content from sources
- Preserve step-by-step explanations, numbered lists, and conceptual frameworks
- Include specific details from proofs, arguments, and analyses when available
- Properly attribute all named persons and their specific contributions  
- Use numbered citations [1], [2], etc. throughout your answer
- Incorporate mathematical, scientific, philosophical, or theological details when present in sources
- When sources contain categorizations or structured breakdowns, include those structures
- Maintain Catholic orthodox perspective
- Be comprehensive and detailed, not just summaries

ENHANCED ANSWER:"""

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                RUNPOD_URL,
                json={
                    "model": "mistralai/Mistral-Small-Instruct-2409",  # Using Mistral-Small 3.1
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1200,  # Increased for detailed responses
                    "temperature": 0.3,  # Lower for accuracy
                    "top_p": 0.9,
                    "stream": False
                },
                headers={"Authorization": f"Bearer {RUNPOD_KEY}"},
                timeout=45.0  # Longer timeout for detailed responses
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("choices") and result["choices"][0].get("message"):
                    return result["choices"][0]["message"]["content"].strip()
                elif result.get("response"):
                    return result["response"].strip()
            
            raise Exception(f"RunPod error: {response.status_code}")
            
    except Exception as e:
        logger.error(f"❌ Answer generation failed: {e}")
        
        # Enhanced fallback with citations
        fallback = f"Based on authoritative Catholic theological sources, particularly from {sources[0].agent.replace('_', ' ')}"
        
        if sources[0].namedPersons:
            fallback += f" (referencing {', '.join(sources[0].namedPersons[:2])})"
        
        fallback += f", this question involves important theological considerations. "
        fallback += f"The sources indicate specific insights from {len(sources)} relevant documents including {sources[0].source}. "
        fallback += "Please try your question again for a detailed response with full citations."
        
        return fallback

@app.post("/ask-simple", response_model=SimplifiedResponse)
async def simplified_query(request: SimplifiedQueryRequest):
    """Simplified query with enhanced citations and proof attribution"""
    
    start_time = time.time()
    logger.info(f"🔍 Simplified query from {request.user}: {request.question}")
    
    try:
        # Enhanced Weaviate search
        sources = await enhanced_weaviate_search(request.question, request.maxChunks)
        
        if not sources:
            return SimplifiedResponse(
                answer="I apologize, but I couldn't find relevant information to answer your question. Please try rephrasing or asking about a different theological topic.",
                sources=[],
                confidence=0.0,
                processingTime=time.time() - start_time,
                citationsUsed=0,
                proofsIncluded=0
            )
        
        # Generate enhanced answer with citations
        answer = await generate_enhanced_answer(
            request.question, 
            sources, 
            request.includeProofs
        )
        
        # Calculate metrics
        avg_distance = sum(s.distance for s in sources) / len(sources)
        confidence = max(0.0, 1.0 - avg_distance)
        citations_used = len([s for s in sources if s.source])
        detailed_content_used = len([s for s in sources if s.hasDetailedContent])
        breakdowns_included = len([s for s in sources if s.hasEnumeration or s.contentType in ["conceptual_breakdown", "step_by_step_explanation"]])
        
        return SimplifiedResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            processingTime=time.time() - start_time,
            citationsUsed=citations_used,
            detailedContentUsed=detailed_content_used,
            breakdownsIncluded=breakdowns_included
        )
        
    except Exception as e:
        logger.error(f"❌ Simplified query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Legacy endpoint for backward compatibility
@app.post("/ask")
async def legacy_query(request: dict):
    """Legacy endpoint - redirects to simplified approach"""
    
    simplified_request = SimplifiedQueryRequest(
        question=request.get("question", ""),
        user=request.get("user", "anonymous")
    )
    
    response = await simplified_query(simplified_request)
    
    # Return simplified format for backward compatibility
    return {
        "answer": response.answer,
        "sources": [{"content": s.content, "source": s.source, "agent": s.agent} for s in response.sources]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
