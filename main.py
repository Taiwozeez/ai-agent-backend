# main.py - FINAL WORKING VERSION
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

# Gemini API - Direct REST call
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def ask_gemini(question: str) -> str:
    """Ask Gemini a question and get answer"""
    if not GOOGLE_API_KEY:
        return None
    
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
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            logger.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error: {e}")
        return None

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
        
        # Try Gemini
        answer = ask_gemini(request.query)
        
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
            # Simple fallback
            return ResearchResponse(
                success=True,
                query=request.query,
                result=f"Here's information about '{request.query}'. The AI service is initializing. Please try again in a moment.",
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
