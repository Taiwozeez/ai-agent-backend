# main.py
import os
from dotenv import load_dotenv
import requests
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import logging
import time
import re

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AI Research Agent API",
    description="Research assistant API powered by Hugging Face with Company Knowledge Base",
    version="2.0.0"
)

# Configure CORS for Next.js - FIXED for Vercel deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://ai-agent-taiwo-adelaja.vercel.app",
        "https://ai-agent-taiwo-adelaja-5a2et6rb0-taiwo-adelajas-projects.vercel.app",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ API CONFIGURATION ============
HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_URL = "https://router.huggingface.co/v1/chat/completions"
HUGGINGFACE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# ============ COMPANY KNOWLEDGE BASE SETUP ============
company_collection = None
try:
    import chromadb
    from chromadb.utils import embedding_functions
    
    CHROMA_PERSIST_DIR = "./company_knowledge_db"
    COMPANY_COLLECTION_NAME = "company_website"
    
    if os.path.exists(CHROMA_PERSIST_DIR):
        try:
            chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            try:
                company_collection = chroma_client.get_collection(name=COMPANY_COLLECTION_NAME)
                logger.info(f"Loaded existing company knowledge base. Contains {company_collection.count()} chunks")
            except:
                embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
                company_collection = chroma_client.get_or_create_collection(
                    name=COMPANY_COLLECTION_NAME,
                    embedding_function=embedding_fn
                )
                logger.info(f"Created collection with embeddings. Contains {company_collection.count()} chunks")
        except Exception as e:
            logger.warning(f"Error loading company knowledge base: {e}")
            company_collection = None
    else:
        logger.info("No company knowledge base found.")
        company_collection = None
        
except Exception as e:
    logger.warning(f"Company knowledge base not available: {e}")
    company_collection = None

# Request/Response Models
class ResearchRequest(BaseModel):
    query: str
    use_fallback: Optional[bool] = True
    use_company_knowledge: Optional[bool] = True

class ResearchResponse(BaseModel):
    success: bool
    query: str
    result: str
    method_used: str
    saved_to_file: bool
    timestamp: str
    sources: Optional[List[str]] = None

class HealthResponse(BaseModel):
    status: str
    api_configured: bool
    company_knowledge_loaded: bool
    knowledge_chunks: int
    timestamp: str

# Helper function to save to file
def save_to_file(data: str) -> str:
    """Save research to a file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = "research_output.txt"
    
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"📝 RESEARCH REPORT\n")
            f.write(f"📅 {timestamp}\n")
            f.write(f"{'='*60}\n")
            f.write(data)
            f.write(f"\n{'='*60}\n")
        return f"✅ Successfully saved to {filename}"
    except Exception as e:
        return f"❌ Error saving: {e}"

# Helper function to detect if question is company-related
def is_company_related(query: str) -> bool:
    """Detect if question is about the company"""
    company_keywords = [
        'sunking', 'sun king', 'solar', 'product', 'service', 'pricing',
        'price', 'cost', 'subscription', 'plan', 'package', 'warranty',
        'support', 'customer service', 'contact', 'return', 'refund',
        'feature', 'installation', 'payment', 'delivery', 'order',
        'company', 'mission', 'vision', 'about', 'founder', 'impact',
        'solar panel', 'home system', 'energy', 'power', 'electricity'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in company_keywords)

# Search company knowledge base
def search_company_knowledge(query: str, top_k: int = 3) -> tuple[str, List[str]]:
    """Search the company website knowledge base"""
    if not company_collection or company_collection.count() == 0:
        return "", []
    
    try:
        results = company_collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        if results and results['documents'] and results['documents'][0]:
            context = "\n\n".join(results['documents'][0])
            sources = []
            if results['metadatas'] and results['metadatas'][0]:
                for meta in results['metadatas'][0]:
                    if meta and 'source' in meta:
                        sources.append(meta['source'])
            return context, list(set(sources))
    except Exception as e:
        logger.error(f"Error searching company knowledge: {e}")
    
    return "", []

# ============ HUGGING FACE API ============
def generate_with_huggingface(prompt: str) -> str:
    """Generate answer using Hugging Face API"""
    if not HF_TOKEN:
        logger.warning("HF_TOKEN not found")
        return None
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": HUGGINGFACE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(HUGGINGFACE_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            logger.warning(f"Hugging Face error: {response.status_code}")
            return None
    except Exception as e:
        logger.warning(f"Hugging Face exception: {e}")
        return None

# ============ MAIN RESEARCH FUNCTION ============
def research_with_api(query: str) -> tuple[str, str]:
    """Research using available APIs"""
    
    prompt = f"""Please provide a clear, informative answer about: {query}
    
Make it easy to understand and well-structured. Keep it concise (2-3 paragraphs max)."""
    
    result = generate_with_huggingface(prompt)
    if result:
        return result, "huggingface"
    
    return "I'm currently unable to connect to the AI service. Please try again in a few moments.", "error"

# ============ API ENDPOINTS ============
@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="healthy",
        api_configured=True,
        company_knowledge_loaded=company_collection is not None and company_collection.count() > 0 if company_collection else False,
        knowledge_chunks=company_collection.count() if company_collection else 0,
        timestamp=datetime.now().isoformat()
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="active",
        api_configured=True,
        company_knowledge_loaded=company_collection is not None and company_collection.count() > 0 if company_collection else False,
        knowledge_chunks=company_collection.count() if company_collection else 0,
        timestamp=datetime.now().isoformat()
    )

@app.post("/api/research", response_model=ResearchResponse)
async def research_endpoint(request: ResearchRequest):
    """Main research endpoint"""
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Research request: {request.query[:50]}...")
        
        result_text = ""
        method_used = ""
        sources = None
        
        # Check if this should use company knowledge
        use_company = request.use_company_knowledge and is_company_related(request.query)
        
        if use_company and company_collection and company_collection.count() > 0:
            logger.info("Using company knowledge base...")
            context, source_urls = search_company_knowledge(request.query)
            
            if context:
                prompt = f"""You are a helpful customer support assistant. Answer based ONLY on the information below.

QUESTION: {request.query}

INFO: {context}

ANSWER:"""
                result_text = generate_with_huggingface(prompt)
                if result_text:
                    method_used = "company_knowledge_base"
                    sources = source_urls
                else:
                    result_text, method_used = research_with_api(request.query)
            else:
                result_text, method_used = research_with_api(request.query)
        else:
            result_text, method_used = research_with_api(request.query)
        
        # Save to file
        save_to_file(f"🔬 RESEARCH TOPIC: {request.query}\n\n{result_text}")
        
        return ResearchResponse(
            success=True,
            query=request.query,
            result=result_text,
            method_used=method_used,
            saved_to_file=True,
            timestamp=datetime.now().isoformat(),
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
