# main.py - Fixed Hugging Face Integration
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
    """Call Hugging Face API with simpler model"""
    if not HF_TOKEN:
        logger.error("HF_TOKEN not set")
        return None
    
    # Try with a simpler, more reliable model
    API_URL = "https://api-inference.huggingface.co/models/gpt2"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
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
        logger.info(f"API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Response type: {type(result)}")
            
            # Parse the response
            if isinstance(result, list) and len(result) > 0:
                text = result[0].get('generated_text', '')
                # Remove the input prompt
                if text.startswith(prompt):
                    text = text[len(prompt):]
                return text.strip()[:300]
            elif isinstance(result, dict) and 'generated_text' in result:
                text = result['generated_text']
                if text.startswith(prompt):
                    text = text[len(prompt):]
                return text.strip()[:300]
        else:
            logger.error(f"API Error {response.status_code}: {response.text[:200]}")
            return None
            
    except Exception as e:
        logger.error(f"Exception: {e}")
        return None
    
    return None

# Test endpoint
@app.get("/test-hf")
async def test_hf():
    if not HF_TOKEN:
        return {"error": "HF_TOKEN not configured", "has_token": False}
    
    test_result = query_huggingface("Hello")
    return {
        "has_token": True,
        "test_result": test_result,
        "working": test_result is not None,
        "message": "Working!" if test_result else "Still failing"
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
            # Informative fallback
            return ResearchResponse(
                success=True,
                query=request.query,
                result=f"I received your question about '{request.query}'. The AI service is currently experiencing high demand. Please try again in a few moments.",
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
