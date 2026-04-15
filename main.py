# main.py - Working Version with CORS
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

# CORS for Vercel - Updated with your URLs
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
    """Call Hugging Face API with proper format"""
    if not HF_TOKEN:
        logger.error("HF_TOKEN not set")
        return None
    
    # Using the correct API endpoint for text generation
    API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 200,
            "temperature": 0.7,
            "do_sample": True,
            "pad_token_id": 50256
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        logger.info(f"Hugging Face response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated = result[0].get('generated_text', '')
                # Remove the input prompt from the response
                if generated.startswith(prompt):
                    generated = generated[len(prompt):]
                return generated.strip()
        else:
            logger.error(f"Hugging Face error: {response.status_code} - {response.text[:200]}")
            return None
    except Exception as e:
        logger.error(f"Exception: {e}")
        return None
    
    return None

# Test endpoint to verify Hugging Face is working
@app.get("/test-hf")
async def test_hf():
    if not HF_TOKEN:
        return {"error": "HF_TOKEN not configured", "has_token": False}
    
    test_result = query_huggingface("Say hello")
    return {
        "has_token": True,
        "token_prefix": HF_TOKEN[:10] + "...",
        "test_result": test_result,
        "working": test_result is not None
    }

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
        
        # Try Hugging Face
        result = query_huggingface(request.query)
        
        if result:
            return ResearchResponse(
                success=True,
                query=request.query,
                result=result,
                method_used="huggingface",
                saved_to_file=False,
                timestamp=datetime.now().isoformat()
            )
        else:
            # Fallback response
            return ResearchResponse(
                success=True,
                query=request.query,
                result=f"Here's information about {request.query}. (Note: AI service is currently unavailable, but this is a fallback response.)",
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
