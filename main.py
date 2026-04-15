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
    description="Research assistant API powered by Hugging Face",
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
HUGGINGFACE_MODEL = "microsoft/DialoGPT-medium"  # Changed to faster model

# ============ COMPANY KNOWLEDGE BASE - DISABLED ============
# Company knowledge is temporarily disabled to prevent memory issues on free tier
company_collection = None
company_knowledge_loaded = False

def get_company_knowledge():
    """Company knowledge disabled - returns None"""
    logger.info("Company knowledge is currently disabled")
    return None

def is_company_related(query: str) -> bool:
    """Company knowledge disabled - always returns False"""
    return False

def search_company_knowledge(query: str, top_k: int = 3):
    """Company knowledge disabled - returns empty"""
    return "", []

# Request/Response Models
class ResearchRequest(BaseModel):
    query: str
    use_fallback: Optional[bool] = True
    use_company_knowledge: Optional[bool] = False  # Disabled by default

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
        "max_tokens": 500,  # Reduced for faster responses
        "temperature": 0.7
    }
    
    try:
        response = requests.post(HUGGINGFACE_URL, headers=headers, json=payload, timeout=30)
        
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
    
    prompt = f"""Please provide a clear, concise answer about: {query}
    
Keep it short (2-3 sentences max)."""
    
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
        company_knowledge_loaded=False,
        knowledge_chunks=0,
        timestamp=datetime.now().isoformat()
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="active",
        api_configured=True,
        company_knowledge_loaded=False,
        knowledge_chunks=0,
        timestamp=datetime.now().isoformat()
    )

@app.post("/api/research", response_model=ResearchResponse)
async def research_endpoint(request: ResearchRequest):
    """Main research endpoint - company knowledge disabled"""
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Research request: {request.query[:50]}...")
        
        # Always use general API (company knowledge disabled)
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
            sources=None
        )
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
