# main.py - With Detailed Error Logging
import os
from dotenv import load_dotenv
import requests
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Tuple
import uvicorn
import logging

load_dotenv()

app = FastAPI(title="AI Research API", version="1.0.0")

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


# ── Models ────────────────────────────────────────────────────────────────────

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
    model: str
    timestamp: str


# ── Config ────────────────────────────────────────────────────────────────────

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# gemini-1.5-flash is shut down — use gemini-2.5-flash
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)


# ── Gemini helper ─────────────────────────────────────────────────────────────

def ask_gemini(question: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Ask Gemini a question.
    Returns (answer, None) on success, or (None, error_message) on failure.
    """
    if not GOOGLE_API_KEY:
        return None, "GOOGLE_API_KEY is not set in environment variables."

    payload = {
        "contents": [{"parts": [{"text": question}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 2048,
        },
    }

    try:
        response = requests.post(
            GEMINI_URL,
            params={"key": GOOGLE_API_KEY},
            json=payload,
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return None, "Gemini returned no candidates."
            answer = candidates[0]["content"]["parts"][0]["text"]
            return answer, None

        # Surface the real error message from Google
        error_body = response.json() if response.content else {}
        error_msg = (
            error_body.get("error", {}).get("message")
            or f"HTTP {response.status_code}: {response.text[:300]}"
        )
        logger.error("Gemini API error: %s", error_msg)
        return None, error_msg

    except requests.exceptions.Timeout:
        msg = "Request to Gemini timed out after 30 seconds."
        logger.error(msg)
        return None, msg
    except Exception as exc:
        logger.error("Gemini exception: %s", exc)
        return None, str(exc)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Simple liveness check."""
    return HealthResponse(
        status="active",
        api_configured=bool(GOOGLE_API_KEY),
        model=GEMINI_MODEL,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/debug")
async def debug():
    """
    Diagnostic endpoint — confirms API key is present and the model responds.
    Remove or protect this endpoint before going to production.
    """
    if not GOOGLE_API_KEY:
        return {
            "status": "error",
            "message": "GOOGLE_API_KEY not found in environment.",
            "fix": "Add GOOGLE_API_KEY in Render Dashboard → Environment Variables.",
        }

    test_result, error = ask_gemini("Reply with the single word: OK")

    return {
        "status": "test_complete",
        "model": GEMINI_MODEL,
        "api_key_configured": True,
        "api_key_prefix": GOOGLE_API_KEY[:15] + "...",
        "gemini_working": test_result is not None,
        "test_result": test_result,
        "error": error,
        "next_steps": (
            []
            if test_result
            else [
                "Your API key may be invalid or the model name has changed.",
                "Check available models: GET https://generativelanguage.googleapis.com"
                "/v1beta/models?key=YOUR_KEY",
                "Current recommended model: gemini-2.5-flash",
                "Update GEMINI_MODEL env var if needed and redeploy.",
            ]
        ),
    }


@app.post("/api/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    """Main research endpoint — forwards the query to Gemini and returns the answer."""
    query = (request.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    logger.info("Researching: %s", query)

    answer, error = ask_gemini(query)

    if answer:
        return ResearchResponse(
            success=True,
            query=query,
            result=answer,
            method_used=GEMINI_MODEL,
            saved_to_file=False,
            timestamp=datetime.now().isoformat(),
        )

    # Return a structured error response instead of raising 500,
    # so the frontend can display a meaningful message.
    logger.error("Gemini failed for query '%s': %s", query, error)
    return ResearchResponse(
        success=False,
        query=query,
        result=f"AI service error: {error}",
        method_used="error",
        saved_to_file=False,
        timestamp=datetime.now().isoformat(),
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
