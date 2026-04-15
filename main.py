# main.py - Without Company Knowledge
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

# Configure CORS for Next.js
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
# Using the correct Inference API endpoint
HUGGINGFACE_URL = "https://api-inference.huggingface.co/models/"
HUGGINGFACE_MODEL = "microsoft/DialoGPT-medium"  # Fast and reliable model

# Request/Response Models
class ResearchRequest(BaseModel):
    query: str
    use_fallback: Optional[bool] = True

class ResearchResponse(BaseModel):
    success: bool
    query: str
    result: str
    method_used: str
    saved_to_file: bool
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    api_configured: bool
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
    """Generate answer using Hugging Face Inference API"""
    if not HF_TOKEN:
        logger.warning("HF_TOKEN not found")
        return None
    
    # Use the correct endpoint for the model
    API_URL = f"{HUGGINGFACE_URL}{HUGGINGFACE_MODEL}"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Format payload correctly for text generation
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "do_sample": True,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=45)
        
        if response.status_code == 200:
            result = response.json()
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                if 'generated_text' in result[0]:
                    return result[0]['generated_text'].strip()
                elif 'content' in result[0]:
                    return result[0]['content'].strip()
            elif isinstance(result, dict):
                if 'generated_text' in result:
                    return result['generated_text'].strip()
                elif 'response' in result:
                    return result['response'].strip()
            
            # Fallback: return the raw result as string
            return str(result)[:500]
        else:
            logger.warning(f"Hugging Face error: {response.status_code} - {response.text[:200]}")
            
            # Try fallback model if main model fails
            if HUGGINGFACE_MODEL != "gpt2":
                logger.info("Trying fallback model: gpt2")
                return generate_with_fallback(prompt)
            return None
    except Exception as e:
        logger.warning(f"Hugging Face exception: {e}")
        return None

def generate_with_fallback(prompt: str) -> str:
    """Fallback to gpt2 model"""
    try:
        API_URL = f"{HUGGINGFACE_URL}gpt2"
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 100,
                "temperature": 0.7,
                "do_sample": True,
                "return_full_text": False
            }
        }
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
                return result[0]['generated_text'].strip()
    except:
        pass
    return None

# ============ MAIN RESEARCH FUNCTION ============
def research_with_api(query: str) -> tuple[str, str]:
    """Research using Hugging Face API"""
    
    prompt = f"Please answer concisely: {query}"
    
    result = generate_with_huggingface(prompt)
    if result:
        # Clean up the response
        result = result.replace(prompt, '').strip()
        if result:
            return result, "huggingface"
    
    return "I'm currently unable to connect to the AI service. Please try again in a few moments.", "error"

# ============ API ENDPOINTS ============
@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="healthy",
        api_configured=True,
        timestamp=datetime.now().isoformat()
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="active",
        api_configured=True,
        timestamp=datetime.now().isoformat()
    )

@app.post("/api/research", response_model=ResearchResponse)
async def research_endpoint(request: ResearchRequest):
    """Main research endpoint"""
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Research request: {request.query[:50]}...")
        
        # Get research result
        result_text, method_used = research_with_api(request.query)
        
        # Save to file
        save_to_file(f"🔬 RESEARCH TOPIC: {request.query}\n\n{result_text}")
        
        return ResearchResponse(
            success=True,
            query=request.query,
            result=result_text,
            method_used=method_used,
            saved_to_file=True,
            timestamp=datetime.now().isoformat()
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
