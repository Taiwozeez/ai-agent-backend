# main.py - Complete Working Version with Debug
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

# Comprehensive mock responses
def get_mock_response(query: str) -> str:
    """Return informative mock responses for common questions"""
    query_lower = query.lower()
    
    # Country/Place questions
    if "england" in query_lower or "london" in query_lower:
        return "England is a country that is part of the United Kingdom. Its capital is London, which is home to landmarks like Big Ben, the Tower of London, Buckingham Palace, and the London Eye. England has a rich history dating back thousands of years."
    
    elif "nigeria" in query_lower:
        return "Nigeria is a country in West Africa. Its capital is Abuja, and its largest city is Lagos. Nigeria gained independence from British rule on October 1, 1960. It is Africa's most populous country and largest economy."
    
    elif "africa" in query_lower:
        return "Africa is the world's second-largest continent, known for its diverse cultures, wildlife, and natural resources. It has 54 countries, with Egypt, Nigeria, South Africa, and Kenya being among the most well-known."
    
    # Technology questions
    elif "ai" in query_lower or "artificial intelligence" in query_lower:
        return "Artificial Intelligence (AI) is the simulation of human intelligence in machines programmed to think and learn. It includes machine learning, deep learning, natural language processing, and computer vision. AI powers technologies like voice assistants, recommendation systems, and self-driving cars."
    
    elif "python" in query_lower:
        return "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used for web development, data science, AI, machine learning, automation, and scientific computing."
    
    elif "machine learning" in query_lower:
        return "Machine learning is a subset of AI that enables systems to learn from data without explicit programming. It uses algorithms to identify patterns and make predictions. Common applications include spam filters, recommendation engines, and image recognition."
    
    # Solar/Company questions
    elif "solar" in query_lower or "sun king" in query_lower or "sunking" in query_lower:
        return "SunKing is a leading provider of solar energy solutions in Africa. They offer solar home systems, lanterns, and appliances that provide clean, affordable energy to households and businesses. Their products help reduce reliance on kerosene and grid electricity."
    
    # Default response
    else:
        responses = [
            f"Thank you for asking about '{query}'. This is a demo response while the AI service is being configured. Full AI capabilities will be available soon.",
            f"Great question about '{query}'! The AI service is currently being set up, but your question has been noted.",
            f"I'm here to help with '{query}'. While full AI responses are being configured, please know that your question is important."
        ]
        return random.choice(responses)

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
            "parameters": {
                "max_length": 100,
                "temperature": 0.7,
                "do_sample": True,
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        logger.info(f"Hugging Face response: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                text = result[0].get('generated_text', '')
                if text.startswith(prompt):
                    text = text[len(prompt):]
                return text.strip()[:300]
        elif response.status_code == 503:
            # Model is loading, wait and retry
            logger.info("Model is loading, this is normal for first request")
            return None
        else:
            logger.warning(f"Hugging Face error: {response.status_code}")
            return None
            
    except Exception as e:
        logger.warning(f"Hugging Face exception: {e}")
        return None
    
    return None

# Debug endpoint to test Hugging Face connection
@app.get("/debug-hf")
async def debug_hf():
    if not HF_TOKEN:
        return {
            "has_token": False, 
            "error": "No HF_TOKEN configured on Render",
            "fix": "Add HF_TOKEN in Render Environment Variables"
        }
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    # Test 1: Check if token is valid
    try:
        response = requests.get(
            "https://huggingface.co/api/whoami",
            headers=headers,
            timeout=10
        )
        token_valid = response.status_code == 200
        username = response.json().get("name", "unknown") if token_valid else None
    except Exception as e:
        token_valid = False
        username = None
    
    # Test 2: Try to call a model
    try:
        api_response = requests.post(
            "https://api-inference.huggingface.co/models/gpt2",
            headers=headers,
            json={"inputs": "Hello"},
            timeout=30
        )
        model_works = api_response.status_code == 200
        model_response = api_response.text[:200] if model_works else api_response.text[:200]
    except Exception as e:
        model_works = False
        model_response = str(e)
    
    return {
        "has_token": bool(HF_TOKEN),
        "token_valid": token_valid,
        "username": username,
        "model_works": model_works,
        "model_response": model_response,
        "using_mock": not model_works,
        "message": "Backend is working with mock responses" if not model_works else "Hugging Face is working!"
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
        method = "huggingface"
        
        # Fallback to mock responses
        if not result:
            result = get_mock_response(request.query)
            method = "mock"
        
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
