#!/usr/bin/env python3
"""
MAGISAI ENHANCED WEAVIATE QUERY AGENT
====================================
Enhanced version with MagisAI persona and post-processing
- Transforms responses to authoritative Catholic voice
- Extracts and formats scholarly citations with footnotes
- Removes AI-like language and adds intellectual rigor
- Based on Fr. Robert Spitzer's research and Catholic teaching
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

app = FastAPI(title="MagisAI Catholic RAG Service", version="4.0.0")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_KEY = os.getenv("WEAVIATE_API_KEY") or os.getenv("WEAVIATE_KEY")

# Initialize Weaviate client and agents
weaviate_client = None
main_query_agent = None

try:
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_KEY)
    )
    
    # Single Query Agent for all functionality
    main_query_agent = QueryAgent(
        client=weaviate_client,
        collections=[
            QueryAgentCollectionConfig(
                name="MagisChunk",
                target_vector=["content_vector"],
            ),
        ],
    )
    
    logger.info("✅ Connected to Weaviate with MagisAI Query Agent")
except Exception as e:
    logger.error(f"❌ Weaviate/Query Agent setup failed: {e}")

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
    answer: str  # Transformed MagisAI response
    suggested_references: Optional[str] = None  # Formatted citations section
    magis_chunk_results: List[MagisChunkResult] = []
    processing_time: float
    method_used: str
    confidence: Optional[float] = None
    # Debug info (optional)
    raw_agent_response: Optional[str] = None
    transformation_notes: Optional[List[str]] = None

@app.get("/")
async def root():
    return {
        "service": "MagisAI Catholic RAG Service",
        "version": "4.0.0",
        "persona": "Authoritative Catholic expert based on Fr. Robert Spitzer's research",
        "features": [
            "magisai_persona_transformation", 
            "scholarly_citation_formatting",
            "catholic_authoritative_voice",
            "fr_spitzer_knowledge_base"
        ],
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "weaviate_connected": weaviate_client is not None and weaviate_client.is_ready(),
        "magisai_agent_ready": main_query_agent is not None,
        "knowledge_base": "Fr. Robert Spitzer's research on science, reason, faith, morality, scripture, and Church history",
        "persona": "MagisAI - Catholic teaching authority"
    }

def clean_ai_language(text: str) -> Tuple[str, List[str]]:
    """Remove AI-like phrases and transform to authoritative Catholic voice"""
    
    transformation_notes = []
    original_text = text
    
    # Document/source reference patterns that undermine authority
    document_patterns = [
        (r"^the documents? (?:and search results )?(?:do not present|present|show|indicate|suggest)", ""),
        (r"^(?:based on )?the (?:sources?|documents?|information) (?:provided|available|presented)", ""),
        (r"^the (?:search results|data|sources?) (?:show|indicate|suggest|reveal)", ""),
        (r"^according to the (?:documents?|sources?|search results)", ""),
        (r"^the (?:available )?information (?:suggests?|indicates?|shows?)", ""),
        (r"^from the (?:sources?|documents?|materials?) (?:provided|available)", ""),
        (r"therefore,? based on the (?:information|sources?) available", "therefore"),
    ]
    
    # Standard AI patterns to remove or replace
    ai_patterns = [
        (r"according to the data", ""),
        (r"from the sources provided", ""),
        (r"some say", ""),
        (r"the system found", ""),
        (r"from my training", ""),
        (r"as an ai", ""),
        (r"i am an ai", ""),
        (r"according to my knowledge", ""),
        (r"based on my understanding", ""),
        (r"the information suggests", ""),
        (r"it appears that", ""),
        (r"it seems that", ""),
        (r"from what i can tell", ""),
        (r"in my analysis", ""),
        (r"the data indicates", ""),
        (r"sources suggest", ""),
        (r"according to sources", ""),
        (r"based on available information", ""),
        (r"the sources argue", ""),
        (r"the documents indicate", ""),
        (r"search results show", ""),
    ]
    
    # Apply document pattern transformations first (these are most critical)
    for pattern, replacement in document_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            transformation_notes.append(f"Removed document reference: '{pattern}'")
    
    # Apply standard AI transformations
    for pattern, replacement in ai_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            transformation_notes.append(f"Removed AI language: '{pattern}'")
    
    # Clean up extra spaces and punctuation
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
    text = re.sub(r'\s+([.,;:])', r'\1', text)  # Space before punctuation
    text = text.strip()
    
    # Add authoritative beginning if text seems weak
    weak_beginnings = [r'^well,', r'^so,', r'^basically,', r'^essentially,', r'^rather,']
    for pattern in weak_beginnings:
        if re.search(pattern, text, re.IGNORECASE):
            text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
            transformation_notes.append(f"Removed weak beginning: '{pattern}'")
    
    # If text starts with uncertainty, make it more definitive
    uncertainty_starters = [
        (r'^(?:it would seem|it appears|it seems) that', ''),
        (r'^(?:the evidence suggests|research indicates) that', ''),
        (r'^(?:one could argue|one might say) that', ''),
    ]
    
    for pattern, replacement in uncertainty_starters:
        if re.search(pattern, text, re.IGNORECASE):
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE).strip()
            transformation_notes.append(f"Removed uncertain starter: '{pattern}'")
    
    return text, transformation_notes

def extract_and_format_citations(magis_chunks: List[MagisChunkResult]) -> Tuple[str, Dict[int, str]]:
    """Extract proper citations and create footnote mapping"""
    
    citations_found = []
    footnote_map = {}
    
    for chunk in magis_chunks:
        # Look for citations in the citations field
        if chunk.citations:
            for citation in chunk.citations:
                if citation and len(citation.strip()) > 10:  # Basic validation
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
    
    # Create footnote mapping
    for i, citation in enumerate(unique_citations, 1):
        footnote_map[i] = citation
    
    # Format the references section
    if unique_citations:
        references_section = "**Suggested References for Further Study:**\n"
        for i, citation in enumerate(unique_citations, 1):
            references_section += f"[{i}] {citation}\n"
        return references_section.strip(), footnote_map
    
    return None, {}

def add_footnote_numbers(text: str, magis_chunks: List[MagisChunkResult], footnote_map: Dict[int, str]) -> str:
    """Add footnote numbers to text where studies or research are mentioned"""
    
    if not footnote_map:
        return text
    
    # Look for patterns that suggest research/studies
    study_patterns = [
        r'(studies? (?:show|demonstrate|indicate|reveal|find))',
        r'(research (?:shows|demonstrates|indicates|reveals|finds))',
        r'(according to (?:studies?|research))',
        r'(peer-reviewed (?:studies?|research))',
        r'(\d+(?:\.\d+)?%)',  # Percentages
        r'(statistical (?:analysis|data|evidence))',
        r'(scientific (?:studies?|research|evidence))',
    ]
    
    footnote_counter = 1
    
    for pattern in study_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        for match in reversed(matches):  # Reverse to maintain positions
            if footnote_counter <= len(footnote_map):
                # Insert footnote number after the match
                end_pos = match.end()
                text = text[:end_pos] + f"[{footnote_counter}]" + text[end_pos:]
                footnote_counter += 1
                if footnote_counter > len(footnote_map):
                    break
    
    return text

def detect_and_format_proofs(text: str, magis_chunks: List[MagisChunkResult]) -> str:
    """Detect when proofs of God are referenced and format them properly"""
    
    # Check if any source files contain the specific proof documents
    proof_sources = {}
    proof_patterns = {
        'thomistic': r'thomistic.*proof',
        'lonerganian': r'lonerganian.*proof', 
        'metaphysical': r'metaphysical.*proof'
    }
    
    for chunk in magis_chunks:
        source_file = chunk.sourceFile.lower()
        if 'thomistic proof' in source_file:
            proof_sources['thomistic'] = chunk.sourceFile
        elif 'lonerganian proof' in source_file:
            proof_sources['lonerganian'] = chunk.sourceFile
        elif 'metaphysical proof' in source_file:
            proof_sources['metaphysical'] = chunk.sourceFile
    
    # Look for proof references in the text and enhance them
    enhanced_text = text
    
    # Patterns that suggest proof references
    proof_reference_patterns = [
        (r'(proof(?:s)? (?:of|for) god)', r'**\1**'),
        (r'(thomistic proof)', r'**Thomistic Proof of God**'),
        (r'(lonerganian proof)', r'**Lonerganian Proof of God**'),
        (r'(metaphysical proof)', r'**Metaphysical Proof of God**'),
        (r'(cosmological argument)', r'**\1**'),
        (r'(ontological argument)', r'**\1**'),
        (r'(teleological argument)', r'**\1**'),
        (r'(first cause)', r'**\1**'),
        (r'(prime mover)', r'**\1**'),
        (r'(necessary being)', r'**\1**'),
    ]
    
    for pattern, replacement in proof_reference_patterns:
        enhanced_text = re.sub(pattern, replacement, enhanced_text, flags=re.IGNORECASE)
    
    # If specific proofs are detected, add structured presentation
    if proof_sources:
        proof_intro = "\n\n**Formal Proofs Available:**\n"
        if 'thomistic' in proof_sources:
            proof_intro += f"• **Thomistic Proof**: Structured step-by-step demonstration\n"
        if 'lonerganian' in proof_sources:
            proof_intro += f"• **Lonerganian Proof**: Comprehensive philosophical argument\n"
        if 'metaphysical' in proof_sources:
            proof_intro += f"• **Metaphysical Proof**: Fundamental ontological reasoning\n"
        
        # Add this if the text discusses proofs but doesn't already have structured format
        if re.search(r'proof', enhanced_text, re.IGNORECASE) and not re.search(r'\*\*.*proof.*\*\*', enhanced_text, re.IGNORECASE):
            enhanced_text += proof_intro
    
    return enhanced_text

def enhance_scholarly_content(text: str, magis_chunks: List[MagisChunkResult]) -> str:
    """Enhance text with scholarly structure when studies/statistics are mentioned"""
    
    enhanced_text = text
    
    # First, detect and format any proofs of God
    enhanced_text = detect_and_format_proofs(enhanced_text, magis_chunks)
    
    # Look for statistical data patterns and enhance them
    stat_patterns = [
        (r'(\d+(?:\.\d+)?%)', r'**\1**'),  # Bold percentages
        (r'(\d+(?:,\d{3})*\s+(?:people|individuals|participants|subjects))', r'**\1**'),  # Bold sample sizes
    ]
    
    for pattern, replacement in stat_patterns:
        enhanced_text = re.sub(pattern, replacement, enhanced_text)
    
    # Look for methodology mentions and structure them
    if re.search(r'(study|research|analysis|survey|investigation)', enhanced_text, re.IGNORECASE):
        # Add structure for scholarly content (if not already present)
        if not re.search(r'\*\*.*?\*\*', enhanced_text):  # No bold formatting yet
            # Look for key findings and emphasize them
            finding_patterns = [
                (r'(found that|discovered that|showed that|demonstrated that)', r'**\1**'),
                (r'(concluded that|determined that|established that)', r'**\1**'),
            ]
            
            for pattern, replacement in finding_patterns:
                enhanced_text = re.sub(pattern, replacement, enhanced_text, flags=re.IGNORECASE)
    
    return enhanced_text

def add_catholic_authority_framing(text: str, question: str) -> str:
    """Add appropriate Catholic teaching authority to responses when needed"""
    
    # If text is too short or already has strong Catholic framing, don't add more
    if len(text) < 50 or re.search(r'the catholic church (?:teaches|holds|maintains)', text, re.IGNORECASE):
        return text
    
    # Detect topics that need Catholic teaching authority
    moral_topics = [
        'abortion', 'euthanasia', 'contraception', 'marriage', 'divorce', 'sexuality',
        'homosexuality', 'gender', 'embryo', 'stem cell', 'cloning', 'suicide',
        'war', 'capital punishment', 'social justice', 'poverty', 'economics'
    ]
    
    theological_topics = [
        'god', 'trinity', 'incarnation', 'salvation', 'grace', 'sin', 'heaven',
        'hell', 'purgatory', 'mary', 'saints', 'prayer', 'mass', 'eucharist',
        'sacraments', 'pope', 'church', 'scripture', 'tradition'
    ]
    
    question_lower = question.lower()
    text_lower = text.lower()
    
    # Determine if this is a moral or theological topic
    is_moral_topic = any(topic in question_lower for topic in moral_topics)
    is_theological_topic = any(topic in question_lower for topic in theological_topics)
    
    # Add appropriate Catholic framing
    if is_moral_topic:
        # For moral issues, emphasize definitive teaching
        if 'abortion' in question_lower:
            if not re.search(r'catholic church|church teaching|magisterium', text_lower):
                text = "The Catholic Church teaches definitively that " + text.lower()
        elif any(topic in question_lower for topic in ['euthanasia', 'suicide', 'killing']):
            if not re.search(r'catholic church|church teaching', text_lower):
                text = "According to Catholic moral teaching, " + text
        else:
            # General moral topics
            if not re.search(r'catholic|church|moral teaching', text_lower):
                text = "Catholic moral theology establishes that " + text
                
    elif is_theological_topic:
        # For theological topics, emphasize doctrine and tradition
        if not re.search(r'catholic|church|doctrine|tradition|scripture', text_lower):
            text = "Catholic doctrine affirms that " + text
    
    # Capitalize first letter after adding prefix
    if text and len(text) > 1:
        # Find the first letter after any added prefix
        match = re.search(r'([a-z])', text)
        if match:
            pos = match.start()
            text = text[:pos] + text[pos].upper() + text[pos+1:]
    
    return text

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

def add_catholic_perspective_emphasis(text: str) -> str:
    """Emphasize Catholic perspective when contrasting views are present"""
    
    # Look for secular vs Catholic distinctions and emphasize Catholic teaching
    if re.search(r'(secular|non-religious|atheist|materialist)', text, re.IGNORECASE):
        # Already has contrasting perspectives, ensure Catholic view is clear
        catholic_phrases = [
            (r'(the church teaches)', r'**the Catholic Church teaches**'),
            (r'(catholic teaching)', r'**Catholic teaching**'),
            (r'(church doctrine)', r'**Church doctrine**'),
            (r'(magisterium)', r'**the Magisterium**'),
        ]
        
        for pattern, replacement in catholic_phrases:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

async def transform_to_magisai_response(raw_response: str, magis_chunks: List[MagisChunkResult], question: str, include_debug: bool = False) -> Dict[str, Any]:
    """Transform raw agent response into authoritative MagisAI format"""
    
    transformation_notes = []
    
    # Step 1: Clean AI language
    cleaned_text, clean_notes = clean_ai_language(raw_response)
    transformation_notes.extend(clean_notes)
    
    # Step 2: Extract and format citations
    references_section, footnote_map = extract_and_format_citations(magis_chunks)
    
    # Step 3: Add footnote numbers to text
    if footnote_map:
        cleaned_text = add_footnote_numbers(cleaned_text, magis_chunks, footnote_map)
        transformation_notes.append(f"Added {len(footnote_map)} footnote references")
    
    # Step 4: Enhance scholarly content
    enhanced_text = enhance_scholarly_content(cleaned_text, magis_chunks)
    
    # Step 5: Add Catholic authority framing and perspective
    authority_framed_text = add_catholic_authority_framing(enhanced_text, question)
    final_text = add_catholic_perspective_emphasis(authority_framed_text)
    
    return {
        "transformed_answer": final_text,
        "references_section": references_section,
        "transformation_notes": transformation_notes if include_debug else None
    }

async def query_with_magisai_transformation(question: str, include_debug: bool = False) -> Dict[str, Any]:
    """Use Weaviate Query Agent and transform to MagisAI persona"""
    
    if not main_query_agent:
        raise HTTPException(status_code=503, detail="MagisAI Agent not available")
    
    try:
        # Get core answer from Weaviate Query Agent
        logger.info("🔍 Running MagisAI query agent...")
        result = main_query_agent.run(question)
        
        # Extract raw answer
        raw_answer = ""
        if hasattr(result, 'final_answer'):
            raw_answer = str(result.final_answer)
        elif hasattr(result, 'answer'):
            raw_answer = str(result.answer)
        else:
            raw_answer = str(result)
        
        logger.info(f"📝 Extracted raw answer: {len(raw_answer)} characters")
        
        # Extract source IDs
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
        
        # Get detailed source content
        detailed_sources = await get_detailed_source_content(source_ids)
        
        # Transform to MagisAI format
        logger.info("🎭 Transforming to MagisAI persona...")
        transformation_result = await transform_to_magisai_response(
            raw_answer, detailed_sources, question, include_debug
        )
        
        logger.info(f"✅ MagisAI transformation completed successfully")
        return {
            "answer": transformation_result["transformed_answer"],
            "references": transformation_result["references_section"],
            "detailed_sources": detailed_sources,
            "confidence": getattr(result, 'confidence', 0.9),
            "raw_response": raw_answer if include_debug else None,
            "transformation_notes": transformation_result.get("transformation_notes")
        }
        
    except Exception as e:
        logger.error(f"❌ MagisAI query failed: {e}")
        raise HTTPException(status_code=500, detail=f"MagisAI processing error: {str(e)}")

@app.post("/ask", response_model=QueryResponse)
async def ask_magisai(request: QueryRequest):
    """Main MagisAI query endpoint with persona transformation"""
    
    start_time = time.time()
    logger.info(f"MagisAI query from {request.user}: {request.question}")
    
    try:
        # Use Query Agent with MagisAI transformation
        result = await query_with_magisai_transformation(
            question=request.question,
            include_debug=request.include_debug_info
        )
        
        return QueryResponse(
            answer=result["answer"],
            suggested_references=result["references"],
            magis_chunk_results=result["detailed_sources"],
            processing_time=time.time() - start_time,
            method_used="magisai_weaviate_agent_with_persona_transformation",
            confidence=result.get("confidence"),
            raw_agent_response=result.get("raw_response"),
            transformation_notes=result.get("transformation_notes")
        )
        
    except Exception as e:
        logger.error(f"❌ MagisAI query failed: {e}")
        
        # Return the enhanced fallback message
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
            raw_agent_response=str(e) if request.include_debug_info else None,
            transformation_notes=["Fallback response due to error"] if request.include_debug_info else None
        )

# Legacy endpoint for backward compatibility
@app.post("/ask-magisai")
async def ask_magisai_legacy(request: QueryRequest):
    """Legacy endpoint - same as /ask"""
    return await ask_magisai(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
