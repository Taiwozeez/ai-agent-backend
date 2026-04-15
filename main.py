# main.py - Complete Fixed Version
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

# CORS for Vercel - Fixed (no wildcards, use explicit or "*" for testing)
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

# Use a proper instruction model (not gpt2)
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

def query_huggingface(prompt: str) -> str:
    """Call Hugging Face API with retry logic and proper formatting"""
    if not HF_TOKEN:
        logger.warning("No HF_TOKEN configured")
        return None
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Format prompt properly for instruction models
    formatted_prompt = f"Answer this question clearly and concisely: {prompt}"
    
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": 150,  # Fixed: use max_new_tokens, not max_length
            "temperature": 0.7,
            "do_sample": True,
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
                timeout=30
            )
            
            logger.info(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated = result[0].get('generated_text', '')
                    # Clean up the response
                    if generated.startswith(formatted_prompt):
                        generated = generated[len(formatted_prompt):]
                    return generated.strip()
                elif isinstance(result, dict) and 'generated_text' in result:
                    return result['generated_text'].strip()
                    
            elif response.status_code == 503:
                logger.warning(f"Model loading (503), attempt {attempt + 1}/3")
                if attempt < 2:  # Don't sleep on last attempt
                    time.sleep(3)  # Wait 3 seconds before retry
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

@app.get("/debug-hf")
async def debug_hf():
    """Debug endpoint to test Hugging Face connection"""
    if not HF_TOKEN:
        return {
            "error": "No HF_TOKEN configured on Render",
            "has_token": False,
            "fix": "Add HF_TOKEN in Render Environment Variables"
        }
    
    # Test the correct endpoint
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
            json={"inputs": "Say hello", "parameters": {"max_new_tokens": 20}},
            timeout=15
        )
        results["tests"].append({
            "status_code": response.status_code,
            "working": response.status_code == 200,
            "response": response.text[:200] if response.status_code != 200 else "Success"
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
            # Informative fallback while AI is initializing
            return ResearchResponse(
                success=True,
                query=request.query,
                result=f"📚 Information about '{request.query}':\n\nThe AI service is currently initializing. Please try again in 10-15 seconds. The first request may take longer as the model loads into memory.",
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
