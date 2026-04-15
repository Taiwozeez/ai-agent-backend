# main.py - Complete Working Version with DialoGPT-small
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
import time

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

# Use a model that works with the free Inference API
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-small"

def query_huggingface(prompt: str) -> str:
    """Call Hugging Face API with retry logic"""
    if not HF_TOKEN:
        logger.warning("No HF_TOKEN configured")
        return None
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Simple payload for DialoGPT
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "do_sample": True,
            "return_full_text": False
        }
    }
    
    # Retry logic for 503 (model loading)
    for attempt in range(3):
        try:
            logger.info(f"Attempt {attempt + 1}: Calling Hugging Face API")
            response = requests.post(
                HUGGINGFACE_API_URL,
                headers=headers,
                json=payload,
                timeout=45
            )
            
            logger.info(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated = result[0].get('generated_text', '')
                    if generated:
                        return generated.strip()
                elif isinstance(result, dict) and 'generated_text' in result:
                    return result['generated_text'].strip()
                else:
                    return str(result)[:200]
                    
            elif response.status_code == 503:
                logger.warning(f"Model loading (503), attempt {attempt + 1}/3")
                if attempt < 2:
                    time.sleep(5)  # Wait 5 seconds before retry
                continue
            else:
                logger.warning(f"API Error {response.status_code}: {response.text[:200]}")
                return None
                
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1}")
            continue
        except Exception as e:
            logger.error(f"Exception on attempt {attempt + 1}: {e}")
            continue
    
    return None

# Comprehensive mock responses for fallback
def get_mock_response(query: str) -> str:
    """Return informative mock responses"""
    query_lower = query.lower()
    
    # Country/Place questions
    if "england" in query_lower:
        return "England is a country that is part of the United Kingdom. Its capital is London, known for landmarks like Big Ben, the Tower of London, and Buckingham Palace."
    elif "nigeria" in query_lower:
        return "Nigeria is a country in West Africa. Its capital is Abuja, and its largest city is Lagos. Nigeria gained independence from British rule on October 1, 1960."
    elif "africa" in query_lower:
        return "Africa is the world's second-largest continent, with 54 countries, diverse cultures, and rich natural resources."
    
    # Technology questions
    elif "ai" in query_lower or "artificial intelligence" in query_lower:
        return "Artificial Intelligence (AI) is the simulation of human intelligence in machines. It includes machine learning, natural language processing, and computer vision."
    elif "python" in query_lower:
        return "Python is a high-level programming language known for its simplicity. It's widely used for web development, data science, and AI."
    
    # Default response
    else:
        return f"Thank you for asking about '{query}'. The AI service is initializing. Please try again in a moment."

@app.get("/debug-hf")
async def debug_hf():
    """Debug endpoint to test Hugging Face connection"""
    if not HF_TOKEN:
        return {
            "error": "No HF_TOKEN configured on Render",
            "has_token": False,
            "fix": "Add HF_TOKEN in Render Environment Variables"
        }
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    results = {
        "has_token": True,
        "token_prefix": HF_TOKEN[:10] + "...",
        "model": HUGGINGFACE_API_URL,
        "tests": []
    }
    
    # Test with a simple request
    try:
        response = requests.post(
            HUGGINGFACE_API_URL,
            headers=headers,
            json={"inputs": "Hello", "parameters": {"max_new_tokens": 20}},
            timeout=30
        )
        results["tests"].append({
            "status_code": response.status_code,
            "working": response.status_code == 200,
            "response": response.text[:200] if response.status_code != 200 else "Success - Model is working!"
        })
    except Exception as e:
        results["tests"].append({"error": str(e), "working": False})
    
    return results

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
        
        # Try Hugging Face first
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
            # Fallback to mock responses
            fallback_result = get_mock_response(request.query)
            return ResearchResponse(
                success=True,
                query=request.query,
                result=fallback_result,
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
