# main.py - With Detailed Error Logging
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
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchRequest(BaseModel):
    query: str

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

# Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def ask_gemini(question: str) -> tuple[str, str]:
    """Ask Gemini a question and return (answer, error_message)"""
    if not GOOGLE_API_KEY:
        return None, "No API key configured"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GOOGLE_API_KEY}"
    
    data = {
        "contents": [{
            "parts": [{"text": question}]
        }]
    }
    
    try:
        response = requests.post(url, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            answer = result['candidates'][0]['content']['parts'][0]['text']
            return answer, None
        else:
            error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            logger.error(f"Gemini API error: {error_msg}")
            return None, error_msg
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Gemini exception: {error_msg}")
        return None, error_msg

@app.get("/debug")
async def debug():
    """Debug endpoint - shows exactly what's wrong"""
    if not GOOGLE_API_KEY:
        return {
            "status": "error",
            "message": "No GOOGLE_API_KEY configured in Render Environment",
            "fix": "Add GOOGLE_API_KEY in Render Dashboard → Environment"
        }
    
    # Test the API key
    test_result, error = ask_gemini("Say 'OK'")
    
    return {
        "status": "test_complete",
        "api_key_configured": True,
        "api_key_prefix": GOOGLE_API_KEY[:15] + "...",
        "gemini_working": test_result is not None,
        "test_result": test_result,
        "error": error,
        "next_steps": [
            "If gemini_working is false, your API key is invalid or expired",
            "Get a new key from: https://makersuite.google.com/app/apikey",
            "Update the key in Render Environment Variables",
            "Redeploy"
        ]
    }

@app.get("/health")
async def health():
    return HealthResponse(
        status="active",
        api_configured=GOOGLE_API_KEY is not None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/api/research")
async def research(request: ResearchRequest):
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Researching: {request.query}")
        
        answer, error = ask_gemini(request.query)
        
        if answer:
            return ResearchResponse(
                success=True,
                query=request.query,
                result=answer,
                method_used="gemini",
                saved_to_file=False,
                timestamp=datetime.now().isoformat()
            )
        else:
            # Return the actual error for debugging
            return ResearchResponse(
                success=True,
                query=request.query,
                result=f"AI Service Error: {error}. Please check the /debug endpoint for details.",
                method_used="error",
                saved_to_file=False,
                timestamp=datetime.now().isoformat()
            )
            
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
