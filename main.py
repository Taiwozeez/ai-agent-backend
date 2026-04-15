# main.py - Gemini Working Version
import os
from dotenv import load_dotenv
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import logging
import google.generativeai as genai

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

# Initialize Gemini with proper configuration
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        # Try different model versions
        models_available = genai.list_models()
        logger.info("Available models:")
        for m in models_available:
            logger.info(f"  {m.name}")
        
        # Use gemini-pro (more stable than 1.5-flash)
        model = genai.GenerativeModel('gemini-pro')
        logger.info("Gemini initialized successfully with gemini-pro")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini: {e}")
        model = None
else:
    model = None
    logger.warning("GOOGLE_API_KEY not configured")

def query_gemini(prompt: str) -> str:
    """Call Gemini API with proper error handling"""
    if not GOOGLE_API_KEY or not model:
        logger.warning("Gemini not configured")
        return None
    
    try:
        # Simple prompt without extra instructions
        response = model.generate_content(prompt)
        
        if response and hasattr(response, 'text') and response.text:
            return response.text.strip()
        elif response and hasattr(response, 'parts'):
            # Alternative response format
            return response.parts[0].text.strip()
        else:
            logger.warning(f"Empty response from Gemini")
            return None
            
    except Exception as e:
        logger.error(f"Gemini error: {e}")
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
    
    # Test with a simple query
    test_result = query_gemini("Say 'Hello World'")
    
    # Also try to list available models
    available_models = []
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
    except:
        pass
    
    return {
        "has_key": True,
        "key_prefix": GOOGLE_API_KEY[:15] + "...",
        "models_available": available_models[:5],
        "test_result": test_result,
        "working": test_result is not None,
        "message": "Gemini is working!" if test_result else "Gemini test failed. Check API key permissions."
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
