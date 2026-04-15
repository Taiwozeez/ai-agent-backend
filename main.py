# main.py - Working with Mock Responses
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
import random

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

# Mock responses for common questions
MOCK_RESPONSES = {
    "what is ai": "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn. It includes machine learning, natural language processing, and computer vision.",
    "what is python": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used for web development, data science, AI, and automation.",
    "machine learning": "Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to find patterns in data.",
    "solar": "Solar energy is power derived from the sun's radiation. It's captured using solar panels and converted into electricity or heat for homes and businesses.",
}

def get_mock_response(query: str) -> str:
    """Return a mock response for common questions"""
    query_lower = query.lower()
    
    # Check for keywords
    for key, response in MOCK_RESPONSES.items():
        if key in query_lower:
            return response
    
    # Default response
    return f"Thank you for asking about '{query}'. This is a demo response while the AI service is being configured. Your question has been noted and full AI capabilities will be available soon."

def query_huggingface(prompt: str) -> str:
    """Try Hugging Face API, fallback to mock"""
    if not HF_TOKEN:
        logger.warning("HF_TOKEN not set, using mock response")
        return None
    
    try:
        API_URL = "https://api-inference.huggingface.co/models/gpt2"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {
            "inputs": prompt,
            "parameters": {"max_length": 80, "temperature": 0.7}
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                text = result[0].get('generated_text', '')
                if text.startswith(prompt):
                    text = text[len(prompt):]
                return text.strip()[:300]
    except Exception as e:
        logger.warning(f"Hugging Face error: {e}")
    
    return None

@app.get("/test-hf")
async def test_hf():
    return {
        "has_token": HF_TOKEN is not None,
        "mode": "Using mock responses - Hugging Face not available",
        "status": "Backend is working correctly"
    }

@app.get("/")
@app.get("/health")
async def health():
    return HealthResponse(
        status="active",
        api_configured=True,
        timestamp=datetime.now().isoformat()
    )

@app.post("/api/research")
async def research_endpoint(request: ResearchRequest):
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Researching: {request.query}")
        
        # Try Hugging Face first
        result = query_huggingface(request.query)
        
        # Fallback to mock responses
        if not result:
            result = get_mock_response(request.query)
            method = "mock"
        else:
            method = "huggingface"
        
        return ResearchResponse(
            success=True,
            query=request.query,
            result=result,
            method_used=method,
            saved_to_file=False,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
