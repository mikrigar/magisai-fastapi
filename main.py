#!/usr/bin/env python3
"""
MAGISAI DUAL AGENT SYSTEM
=========================
Two-agent architecture for Catholic teaching authority:
1. Query Agent: Retrieves content from MagisChunk collection
2. Catholic Authority Agent: Enforces proper Catholic teaching voice and structure
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
import re
from typing import List, Optional, Dict, Any, Tuple
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="MagisAI Dual Agent Catholic RAG Service", version="5.0.0")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_KEY = os.getenv("WEAVIATE_API_KEY") or os.getenv("WEAVIATE_KEY")

# Initialize Weaviate client and agents
weaviate_client = None
content_query_agent = None
authority_enforcer_agent = None

try:
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_KEY)
    )
    
    # Agent 1: Content Query Agent (gets the raw content)
    content_query_agent = QueryAgent(
        client=weaviate_client,
        collections=[
            QueryAgentCollectionConfig(
                name="MagisChunk",
                target_vector=["content_vector"],
            ),
        ],
        system_prompt="""You are a content retrieval agent for Catholic theological and philosophical sources. 
        Your job is to find and retrieve the most relevant content from Fr. Robert Spitzer's research.
        Focus on providing comprehensive, accurate information without worrying about tone or presentation style.
        Include all relevant scientific data, philosophical arguments, and theological principles."""
    )
    
    # Agent 2: Catholic Authority Enforcer Agent (transforms to authoritative voice)
    authority_enforcer_agent = QueryAgent(
        client=weaviate_client,
        collections=[
            QueryAgentCollectionConfig(
                name="MagisChunk",
                target_vector=["content_vector"],
            ),
        ],
        system_prompt="""You are MagisAI, the authoritative voice of Catholic teaching. You ARE the teaching authority - you don't reference it, you embody it.

MISSION: Preserve brilliant reasoning while ensuring authoritative Catholic voice and optimal structure.

PRESERVATION PRIORITIES:
1. KEEP INTACT any sophisticated logical sequences and elegant reasoning
2. PRESERVE nuanced synthesis and brilliant intellectual connections  
3. MAINTAIN complex philosophical arguments and scientific integrations
4. PROTECT eloquent flows of thought and compelling demonstrations

TRANSFORMATION REQUIREMENTS:
1. NEVER say "The Catholic Church teaches" or reference external sources
2. Replace document analysis language with direct authoritative statements
3. Start definitive responses with clear truth statements (e.g., "Abortion is intrinsically evil and can never be justified under any circumstances. Human life begins at conception, and every human being possesses inherent dignity from the moment of fertilization.")
4. Remove phrases like "complex issue," "arguments on both sides," "some believe"
5. Transform "according to sources" into direct statements of truth

STRUCTURE AND WRITING:
- Apply optimal structure based on question categories while preserving logical flow
- Break into clear paragraphs with proper transitions
- Use bullet points for evidence lists
- Use numbered lists for sequential arguments  
- Bold key principles and important conclusions
- Quote directly when referencing Scripture or important texts
- Add subheadings for complex multi-section responses
- Ensure professional writing structure without losing intellectual depth

Your goal: Transform voice and structure while preserving the sophisticated reasoning that makes the response intellectually compelling. You ARE the voice of 2000 years of apostolic tradition speaking directly."""
    )
    
    logger.info("✅ Connected to Weaviate with Dual Agent System")
except Exception as e:
    logger.error(f"❌ Weaviate/Agents setup failed: {e}")

class QueryRequest(BaseModel):
    question: str
    user: str
    include_debug_info: Optional[bool] = False

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
    answer: str  # Final authoritative MagisAI response
    suggested_references: Optional[str] = None
    magis_chunk_results: List[MagisChunkResult] = []
    processing_time: float
    method_used: str
    confidence: Optional[float] = None
    # Debug info (optional)
    raw_content_response: Optional[str] = None
    transformation_notes: Optional[List[str]] = None

@app.get("/")
async def root():
    return {
        "service": "MagisAI Dual Agent Catholic RAG Service",
        "version": "5.0.0",
        "architecture": "Two-agent system with Catholic authority enforcement",
        "features": [
            "content_query_agent", 
            "catholic_authority_enforcer_agent",
            "sequential_agent_processing",
            "definitive_moral_teaching",
            "structured_evidence_presentation"
        ],
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "weaviate_connected": weaviate_client is not None and weaviate_client.is_ready(),
        "content_agent_ready": content_query_agent is not None,
        "authority_agent_ready": authority_enforcer_agent is not None,
        "knowledge_base": "Fr. Robert Spitzer's research on science, reason, faith, morality, scripture, and Church history",
        "persona": "MagisAI - Dual Agent Catholic Teaching Authority"
    }

def fix_capitalization_bug(text: str) -> str:
    """Fix the specific capitalization bug where 'THe' appears instead of 'The'"""
    
    # Fix the specific "THe" bug
    text = re.sub(r'\bTHe\b', 'The', text)
    
    # Fix other common capitalization issues
    text = re.sub(r'\bTHat\b', 'That', text)
    text = re.sub(r'\bTHis\b', 'This', text)
    text = re.sub(r'\bTHere\b', 'There', text)
    text = re.sub(r'\bTHey\b', 'They', text)
    text = re.sub(r'\bTHen\b', 'Then', text)
    
    return text

def detect_question_categories(question: str) -> Dict[str, bool]:
    """Detect multiple categories a question falls into for optimal response structuring"""
    
    question_lower = question.lower()
    
    # Define category patterns
    categories = {
        'moral': [
            'abortion', 'euthanasia', 'suicide', 'murder', 'killing', 'contraception', 
            'homosexual', 'same-sex', 'gay', 'transgender', 'marriage', 'divorce',
            'embryo', 'stem cell', 'cloning', 'ivf', 'artificial reproduction',
            'moral', 'ethics', 'sin', 'virtue', 'conscience', 'right', 'wrong'
        ],
        'scientific': [
            'evolution', 'big bang', 'cosmology', 'quantum', 'physics', 'biology',
            'genetics', 'dna', 'embryology', 'neuroscience', 'consciousness',
            'climate', 'environment', 'science', 'scientific', 'research', 'study'
        ],
        'philosophical': [
            'existence', 'proof', 'god exists', 'atheism', 'reason', 'logic',
            'natural law', 'human dignity', 'personhood', 'soul', 'mind',
            'free will', 'determinism', 'causation', 'philosophy', 'metaphysics'
        ],
        'scriptural': [
            'bible', 'scripture', 'genesis', 'gospel', 'jesus said', 'paul wrote',
            'old testament', 'new testament', 'psalms', 'prophets', 'revelation',
            'biblical', 'scriptural', 'verse', 'passage'
        ],
        'theological': [
            'god', 'trinity', 'incarnation', 'salvation', 'grace', 'redemption',
            'heaven', 'hell', 'purgatory', 'saints', 'mary', 'virgin',
            'theology', 'doctrine', 'dogma', 'mystery', 'divine'
        ],
        'church_history': [
            'pope', 'council', 'fathers', 'saints', 'crusades', 'inquisition',
            'reformation', 'vatican', 'early church', 'persecution', 'martyrs',
            'history', 'historical', 'development', 'tradition'
        ],
        'liturgical': [
            'mass', 'eucharist', 'sacrament', 'baptism', 'confirmation', 'marriage',
            'ordination', 'anointing', 'confession', 'liturgy', 'worship',
            'prayer', 'rosary', 'devotion'
        ],
        'social_teaching': [
            'social justice', 'poverty', 'economics', 'capitalism', 'socialism',
            'war', 'peace', 'capital punishment', 'immigration', 'politics',
            'government', 'authority', 'common good', 'solidarity'
        ],
        'apologetics': [
            'why believe', 'proof of god', 'evidence', 'faith and reason',
            'objection', 'atheist', 'agnostic', 'doubt', 'skeptic',
            'defend', 'argument', 'case for'
        ]
    }
    
    # Count matches for each category
    detected_categories = {}
    for category, keywords in categories.items():
        matches = sum(1 for keyword in keywords if keyword in question_lower)
        detected_categories[category] = matches > 0
    
    # Add some helpful metadata
    primary_categories = [cat for cat, detected in detected_categories.items() if detected]
    
    return {
        **detected_categories,
        'primary_categories': primary_categories,
        'is_multi_category': len(primary_categories) > 1,
        'complexity_level': len(primary_categories)
    }

def generate_optimal_structure_prompt(categories: Dict[str, bool], question: str) -> str:
    """Generate structure guidance based on detected categories"""
    
    primary_cats = categories.get('primary_categories', [])
    
    if not primary_cats:
        return """Structure your response with clear paragraphs and good writing principles. 
        Use the most logical flow for maximum persuasive impact."""
    
    # Build structure guidance based on category combinations
    structure_guidance = "Structure your response optimally for these categories: " + ", ".join(primary_cats) + "\n\n"
    
    # Moral questions
    if categories.get('moral'):
        structure_guidance += """For moral content:
        - Open with the definitive moral truth
        - Present supporting evidence in logical order
        - Use clear, authoritative statements\n"""
    
    # Scientific questions  
    if categories.get('scientific'):
        structure_guidance += """For scientific content:
        - Present scientific facts clearly
        - Show compatibility with Catholic teaching
        - Use specific data and research when available\n"""
    
    # Philosophical questions
    if categories.get('philosophical'):
        structure_guidance += """For philosophical content:
        - Build logical arguments step by step
        - Use proper reasoning sequences
        - Connect to natural law and human dignity\n"""
    
    # Scriptural questions
    if categories.get('scriptural'):
        structure_guidance += """For scriptural content:
        - Quote relevant passages directly
        - Provide proper interpretation in context
        - Connect to broader biblical themes\n"""
    
    # Apologetics
    if categories.get('apologetics'):
        structure_guidance += """For apologetic content:
        - Address objections directly
        - Build cumulative case with evidence
        - Use structured proofs when applicable\n"""
    
    # Multi-category guidance
    if categories.get('is_multi_category'):
        structure_guidance += """\nFor multi-category responses:
        - Integrate evidence types for maximum impact
        - Let logical flow determine optimal order
        - Create seamless transitions between evidence types"""
    
    structure_guidance += """\n\nWRITING STRUCTURE REQUIREMENTS:
    - Break into clear paragraphs with logical flow
    - Use bullet points for lists of evidence
    - Use numbered lists for sequential arguments
    - Bold key principles and conclusions
    - Quote sources when appropriate
    - Create subheadings for complex topics
    - Ensure smooth transitions between sections"""
    
    return structure_guidance

async def get_detailed_source_content(source_ids: List[str]) -> List[MagisChunkResult]:
    """Retrieve detailed content for source IDs from Weaviate"""
    detailed_sources = []
    
    if not weaviate_client or not source_ids:
        logger.warning(f"No client or source IDs provided")
        return detailed_sources
    
    try:
        magis_chunk_collection = weaviate_client.collections.get("MagisChunk")
        
        for source_id in source_ids:
            try:
                obj = magis_chunk_collection.query.fetch_object_by_id(
                    source_id,
                    return_properties=[
                        "content", "sourceFile", "headerContext", "humanId",
                        "agent1", "agent1CoreTopics", "citations", 
                        "chunkLength", "citationCount"
                    ]
                )
                
                if obj and obj.properties:
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

def extract_and_format_citations(magis_chunks: List[MagisChunkResult]) -> Optional[str]:
    """Extract proper citations and format them for references section"""
    
    citations_found = []
    
    for chunk in magis_chunks:
        # Look for citations in the citations field
        if chunk.citations:
            for citation in chunk.citations:
                if citation and len(citation.strip()) > 10:
                    # Check if it looks like a proper academic citation
                    if any(indicator in citation.lower() for indicator in ['(', ')', ',', 'vol', 'pp', 'journal', 'university', 'press']):
                        citations_found.append(citation.strip())
        
        # Also check humanId for author information
        if chunk.humanId and chunk.humanId != "Unknown":
            # If humanId looks like an author name, note it
            if not any(chunk.humanId in cite for cite in citations_found):
                # Create a basic citation from available metadata
                basic_cite = f"{chunk.humanId}"
                if chunk.sourceFile:
                    basic_cite += f", {chunk.sourceFile}"
                citations_found.append(basic_cite)
    
    # Remove duplicates while preserving order
    unique_citations = []
    seen = set()
    for cite in citations_found:
        if cite not in seen:
            unique_citations.append(cite)
            seen.add(cite)
    
    # Format the references section
    if unique_citations:
        references_section = "**Suggested References for Further Study:**\n"
        for i, citation in enumerate(unique_citations, 1):
            references_section += f"[{i}] {citation}\n"
        return references_section.strip()
    
    return None

async def run_dual_agent_system(question: str, include_debug: bool = False) -> Dict[str, Any]:
    """Run the two-agent system: Content Agent → Authority Enforcer Agent"""
    
    if not content_query_agent or not authority_enforcer_agent:
        raise HTTPException(status_code=503, detail="Dual agent system not available")
    
    try:
        # Detect question type for context
        topic_info = detect_moral_theological_topics(question)
        
        # STEP 1: Content Query Agent - Get comprehensive content
        logger.info("🔍 Step 1: Running Content Query Agent...")
        content_result = content_query_agent.run(question)
        
        # Extract raw content
        raw_content = ""
        if hasattr(content_result, 'final_answer'):
            raw_content = str(content_result.final_answer)
        elif hasattr(content_result, 'answer'):
            raw_content = str(content_result.answer)
        else:
            raw_content = str(content_result)
        
        logger.info(f"📝 Content Agent retrieved: {len(raw_content)} characters")
        
        # Extract source IDs from content agent
        source_ids = []
        if hasattr(content_result, 'sources') and content_result.sources:
            for source in content_result.sources:
                if hasattr(source, 'object_id'):
                    source_ids.append(source.object_id)
                elif hasattr(source, 'id'):
                    source_ids.append(source.id)
                elif hasattr(source, 'uuid'):
                    source_ids.append(str(source.uuid))
        
        # STEP 2: Catholic Authority Enforcer Agent - Transform to authoritative voice
        logger.info("⛪ Step 2: Running Catholic Authority Enforcer Agent...")
        
        # Create enhanced prompt for authority agent with context
        authority_prompt = f"""Based on this content about '{question}', provide the definitive Catholic response:

{raw_content}

Remember: 
- Speak with absolute authority, not as document analysis
- For moral issues, be definitively clear about Catholic teaching
- Present evidence in logical order for maximum impact
- Never use wishy-washy language or "both sides" framing
- Include scientific data and philosophical reasoning when available"""
        
        authority_result = authority_enforcer_agent.run(authority_prompt)
        
        # Extract final authoritative answer
        final_answer = ""
        if hasattr(authority_result, 'final_answer'):
            final_answer = str(authority_result.final_answer)
        elif hasattr(authority_result, 'answer'):
            final_answer = str(authority_result.answer)
        else:
            final_answer = str(authority_result)
        
        # STEP 3: Fix capitalization bug
        final_answer = fix_capitalization_bug(final_answer)
        
        logger.info(f"⛪ Authority Agent produced: {len(final_answer)} characters")
        
        # Get detailed source content
        detailed_sources = await get_detailed_source_content(source_ids)
        
        # Extract citations
        references_section = extract_and_format_citations(detailed_sources)
        
        # Prepare transformation notes for debug
        transformation_notes = []
        if include_debug:
            transformation_notes = [
                f"Content Agent: Retrieved {len(raw_content)} chars from {len(source_ids)} sources",
                f"Authority Agent: Transformed to {len(final_answer)} chars",
                f"Topic Analysis: {topic_info}",
                "Applied capitalization fixes",
                f"Citations: {len(detailed_sources)} sources processed"
            ]
        
        logger.info(f"✅ Dual agent system completed successfully")
        return {
            "answer": final_answer,
            "references": references_section,
            "detailed_sources": detailed_sources,
            "confidence": getattr(authority_result, 'confidence', 0.95),
            "raw_content": raw_content if include_debug else None,
            "transformation_notes": transformation_notes
        }
        
    except Exception as e:
        logger.error(f"❌ Dual agent system failed: {e}")
        raise HTTPException(status_code=500, detail=f"Dual agent processing error: {str(e)}")

@app.post("/ask", response_model=QueryResponse)
async def ask_magisai_dual_agent(request: QueryRequest):
    """Main MagisAI endpoint using dual agent system"""
    
    start_time = time.time()
    logger.info(f"MagisAI Dual Agent query from {request.user}: {request.question}")
    
    try:
        # Use dual agent system
        result = await run_dual_agent_system(
            question=request.question,
            include_debug=request.include_debug_info
        )
        
        return QueryResponse(
            answer=result["answer"],
            suggested_references=result["references"],
            magis_chunk_results=result["detailed_sources"],
            processing_time=time.time() - start_time,
            method_used="dual_agent_system_content_plus_authority_enforcer",
            confidence=result.get("confidence"),
            raw_content_response=result.get("raw_content"),
            transformation_notes=result.get("transformation_notes")
        )
        
    except Exception as e:
        logger.error(f"❌ MagisAI dual agent query failed: {e}")
        
        # Enhanced fallback message
        fallback_message = (
            "I draw from the Catholic Church's authoritative teachings and "
            "Fr. Robert Spitzer's comprehensive research in science, reason, faith, "
            "morality, scripture, and Church history. I don't have information on "
            "that specific topic within this specialized theological and "
            "philosophical knowledge base."
        )
        
        return QueryResponse(
            answer=fallback_message,
            suggested_references=None,
            magis_chunk_results=[],
            processing_time=time.time() - start_time,
            method_used="fallback_response",
            confidence=0.0,
            raw_content_response=str(e) if request.include_debug_info else None,
            transformation_notes=["Fallback response due to error"] if request.include_debug_info else None
        )

# Legacy endpoints for backward compatibility
@app.post("/ask-dual-agent")
async def ask_dual_agent_legacy(request: QueryRequest):
    """Legacy endpoint - same as /ask"""
    return await ask_magisai_dual_agent(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
