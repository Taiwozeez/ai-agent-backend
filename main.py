# main.py - Gemini with correct model names
import os
from dotenv import load_dotenv
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import logging
from google import genai

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

# Gemini Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the Gemini client
if GOOGLE_API_KEY:
    client = genai.Client(api_key=GOOGLE_API_KEY)
else:
    client = None
    logger.warning("GOOGLE_API_KEY not configured")

def query_gemini(prompt: str) -> str:
    """Call Gemini API with correct model name"""
    if not GOOGLE_API_KEY or not client:
        logger.warning("Gemini not configured")
        return None
    
    # Try different model names that work with free tier
    models_to_try = [
        "models/gemini-1.5-flash",  # Most common free model
        "gemini-1.5-flash",
        "models/gemini-pro",
        "gemini-pro",
    ]
    
    for model_name in models_to_try:
        try:
            logger.info(f"Trying model: {model_name}")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "temperature": 0.7,
                    "max_output_tokens": 200,
                }
            )
            
            if response and response.text:
                logger.info(f"Success with model: {model_name}")
                return response.text.strip()
        except Exception as e:
            logger.warning(f"Model {model_name} failed: {e}")
            continue
    
    logger.error("All Gemini models failed")
    return None

def get_mock_response(query: str) -> str:
    """Fallback mock responses"""
    query_lower = query.lower()
    
    if "england" in query_lower:
        return "England is a country that is part of the United Kingdom. Its capital is London, known for landmarks like Big Ben and Buckingham Palace."
    elif "nigeria" in query_lower:
        return "Nigeria is a country in West Africa. Its capital is Abuja, and its largest city is Lagos. Nigeria gained independence on October 1, 1960."
    elif "ai" in query_lower or "artificial intelligence" in query_lower:
        return "Artificial Intelligence (AI) is the simulation of human intelligence in machines. It includes machine learning, natural language processing, and computer vision."
    else:
        return f"Here's information about '{query}'. The AI service is ready - please try again!"

@app.get("/debug")
async def debug():
    """Debug endpoint to test Gemini connection"""
    if not GOOGLE_API_KEY:
        return {
            "error": "No GOOGLE_API_KEY configured on Render",
            "has_key": False,
            "fix": "Add GOOGLE_API_KEY in Render Environment Variables"
        }
    
    # Try to list available models
    available_models = []
    try:
        # This may not work with all API keys
        for model in client.models.list():
            available_models.append(model.name)
    except Exception as e:
        available_models = [f"Could not list models: {e}"]
    
    # Test with a simple query using the correct format
    test_result = query_gemini("Say 'Hello World'")
    
    return {
        "has_key": True,
        "key_prefix": GOOGLE_API_KEY[:15] + "...",
        "library": "google.genai",
        "available_models": available_models[:5],
        "test_result": test_result,
        "working": test_result is not None,
        "message": "Gemini is working!" if test_result else "Gemini test failed. Check your API key permissions."
    }

@app.get("/")
@app.get("/health")
async def health():
    return HealthResponse(
        status="active",
        api_configured=GOOGLE_API_KEY is not None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/api/research")
async def research_endpoint(request: ResearchRequest):
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Researching: {request.query}")
        
        # Try Gemini first
        result = query_gemini(request.query)
        
        if result:
            return ResearchResponse(
                success=True,
                query=request.query,
                result=result,
                method_used="gemini",
                saved_to_file=False,
                timestamp=datetime.now().isoformat()
            )
        else:
            # Fallback to mock
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
