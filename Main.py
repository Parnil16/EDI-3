import os
import threading
import asyncio
from typing import Dict
from transformers import pipeline
from langchain_community.llms import LlamaCpp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx

from Neural_net.synthia_core import SynthiaCore

print("Loading models... This may take a few minutes...")

try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model=os.getenv("SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
    )
    emotion_analyzer = pipeline(
        "text-classification",
        model=os.getenv("EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")
    )

    model_path = os.getenv("LLM_MODEL_PATH", os.path.join(os.path.dirname(__file__), "Models", "Phi-3-mini-4k-instruct.Q4_0.gguf"))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LLM model not found at {model_path}")

    gpu_layers = int(os.getenv("LLM_GPU_LAYERS", "-1"))
    try:
        import llama_cpp
    except ImportError:
        gpu_layers = 0

    llm = LlamaCpp(model_path=model_path, n_gpu_layers=gpu_layers, n_ctx=4096, verbose=False)
    llm_lock = threading.Lock()
    print("✅ Models loaded successfully! Ready for conversation.")
except Exception as e:
    raise RuntimeError(f"Error loading models: {e}")

# --- Per-user bot management ---
user_bots: Dict[str, SynthiaCore] = {}

def get_bot(user_id: str) -> SynthiaCore:
    if user_id not in user_bots:
        print(f"Creating new bot for user {user_id}")
        user_bots[user_id] = SynthiaCore(
            user_id=user_id,
            llm=llm,
            llm_lock=llm_lock,
            sentiment_analyzer=sentiment_analyzer,
            emotion_analyzer=emotion_analyzer
        )
    return user_bots[user_id]

# --- FastAPI app setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class FeedbackRequest(BaseModel):
    label: str

# --- Send data to frontend ---
async def send_to_localhost(data: dict):
    webapp_url = "https://intercommunicative-squashier-terrence.ngrok-free.dev/incoming"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(webapp_url, json=data)
            if resp.status_code != 200:
                print(f"[WARN] Webapp responded {resp.status_code}: {resp.text}")
            else:
                print("[INFO] Forwarded response to webapp successfully.")
    except Exception as e:
        print(f"[ERROR] Could not send data to webapp: {e}")

# --- Chat endpoint ---
@app.post("/chat/{user_id}")
async def chat_endpoint(user_id: str, req: ChatRequest):
    if len(req.message) > 2000:
        raise HTTPException(status_code=413, detail="Message too long")

    bot = get_bot(user_id)
    reply, emotions, predicted_mood, confidence = await asyncio.to_thread(bot.generate_response, req.message)

    response_data = {
        "user_id": user_id,
        "reply": reply,
        "emotions": emotions,
        "predicted_mood": predicted_mood,
        "confidence": confidence
    }

    await send_to_localhost(response_data)
    headers = {"ngrok-skip-browser-warning": "true"}
    return JSONResponse(content=response_data, headers=headers)

@app.post("/feedback/{user_id}")
async def feedback_endpoint(user_id: str, req: FeedbackRequest):
    bot = get_bot(user_id)
    if req.label not in bot.personalizer.mood_labels:
        raise HTTPException(status_code=400, detail="Invalid label")
    await asyncio.to_thread(bot.record_mood_feedback, req.label)
    return {"status": "feedback recorded", "user": user_id, "label": req.label}

@app.get("/ping")
def ping():
    return {"status": "API running"}

# --- Main run ---
if __name__ == "__main__":
    import uvicorn
    from pyngrok import ngrok

    try:
        ngrok.set_auth_token("353JGfpPw5vfnhjsiuc8W5YAWBt_84b4naxP8FzUoGgG8feEH")
        tunnel = ngrok.connect(8000)
        print(f"[INFO] Ngrok tunnel opened at: {tunnel.public_url}")
    except Exception as e:
        print(f"[ERROR] Failed to start ngrok: {e}")
        tunnel = None

    print("✅ Synthia AI backend running on localhost:8000")
    if tunnel:
        print(f"🌍 Public endpoint: {tunnel.public_url}")

    uvicorn.run(app, host="127.0.0.1", port=8000)