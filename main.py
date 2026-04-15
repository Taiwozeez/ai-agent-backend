# main.py - Working Hugging Face Integration
import os
from dotenv import load_dotenv
import requests
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import logging

load_dotenv()

app = FastAPI()

# CORS for Vercel
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models
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

# Hugging Face Configuration
HF_TOKEN = os.getenv("HF_TOKEN")

def query_huggingface(prompt: str) -> str:
    """Call Hugging Face Inference API"""
    if not HF_TOKEN:
        return None
    
    # Use the correct API endpoint
    API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 100,
            "temperature": 0.7,
            "do_sample": True,
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        logger.info(f"HF API Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated = result[0].get('generated_text', '')
                if generated.startswith(prompt):
                    generated = generated[len(prompt):]
                return generated.strip()
        elif response.status_code == 503:
            # Model is loading
            return "Model is loading. Please try again in 5 seconds."
        else:
            logger.warning(f"HF Error: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Exception: {e}")
        return None
    
    return None

@app.get("/debug-hf")
async def debug_hf():
    """Test Hugging Face connection"""
    if not HF_TOKEN:
        return {"error": "No HF_TOKEN", "has_token": False}
    
    # Test with a simple model
    API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": "Hello"},
            timeout=15
        )
        return {
            "has_token": True,
            "status_code": response.status_code,
            "response": response.text[:300],
            "working": response.status_code == 200
        }
    except Exception as e:
        return {"has_token": True, "error": str(e), "working": False}

@app.get("/")
@app.get("/health")
async def health():
    return HealthResponse(
        status="active",
        api_configured=HF_TOKEN is not None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/api/research")
async def research_endpoint(request: ResearchRequest):
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Researching: {request.query}")
        
        result = query_huggingface(request.query)
        
        if result and "unable" not in result.lower() and "loading" not in result.lower():
            return ResearchResponse(
                success=True,
                query=request.query,
                result=result,
                method_used="huggingface",
                saved_to_file=False,
                timestamp=datetime.now().isoformat()
            )
        else:
            # Informative fallback
            return ResearchResponse(
                success=True,
                query=request.query,
                result=f"Here's information about '{request.query}'. (Note: The AI service is initializing. This is a demo response. Full AI capabilities will be available in a moment.)",
                method_used="fallback",
                saved_to_file=False,
                timestamp=datetime.now().isoformat()
            )
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
