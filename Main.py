# main.py
import os
import sys
import threading  # <-- Imported threading
from transformers import pipeline
from langchain_community.llms import LlamaCpp
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import HTTPException
import asyncio
from typing import Dict

# --- FIX 3: Corrected import path ---
from Neural_net.synthia_core import SynthiaCore

print("Loading models... This may take a few minutes.")

# --- Initialize LLMs (Globally) ---
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model=os.getenv("SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
    )
    emotion_analyzer = pipeline(
        "text-classification",
        model=os.getenv("EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")
    )

    model_path = os.getenv("LLM_MODEL_PATH", os.path.join(os.path.dirname(__file__), "Models", "Meta-Llama-3-8B-Instruct.Q4_0.gguf"))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LLM model file not found at {model_path}. Set LLM_MODEL_PATH env var or place model at ./Models/")

    gpu_layers = int(os.getenv("LLM_GPU_LAYERS", "-1"))
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        gpu_layers = 0

    llm = LlamaCpp(model_path=model_path, n_gpu_layers=gpu_layers, n_ctx=4096, verbose=False)
    
    # --- FIX 1 (Merge): Create the global lock ---
    llm_lock = threading.Lock()
    
    print("Models loaded successfully! Ready for conversation.")
except Exception as e:
    raise RuntimeError(f"Error loading models: {e}")


# --- FIX 1 (Merge): Per-User Bot Management ---
user_bots: Dict[str, SynthiaCore] = {}

def get_bot(user_id: str) -> SynthiaCore:
    """Gets or creates a bot instance for a user."""
    if user_id not in user_bots:
        print(f"Creating new bot instance for user: {user_id}")
        user_bots[user_id] = SynthiaCore(
            user_id=user_id,
            llm=llm,
            llm_lock=llm_lock,  # <-- Pass the lock
            sentiment_analyzer=sentiment_analyzer,
            emotion_analyzer=emotion_analyzer
        )
    return user_bots[user_id]
# --- End Fix 1 ---


# --- FastAPI app ---
app = FastAPI()

class ChatRequest(BaseModel):
    message: str

# --- FIX 6: Add Feedback Request Model ---
class FeedbackRequest(BaseModel):
    label: str  # "calm", "stressed", or "happy"
# --- End Fix 6 ---


@app.post("/chat/{user_id}") # <-- FIX 1: Add user_id
async def chat_endpoint(user_id: str, req: ChatRequest):
    if len(req.message) > 2000:
        raise HTTPException(status_code=413, detail="Message too long")
    
    bot = get_bot(user_id) # <-- FIX 1: Get user-specific bot
    
    # Use asyncio.to_thread to run the blocking function in a separate thread
    reply, emotions, predicted_mood, confidence = await asyncio.to_thread(bot.generate_response, req.message)
    return {"reply": reply, "emotions": emotions, "predicted_mood": predicted_mood, "confidence": confidence}

# --- FIX 6: Add Feedback Endpoint ---
@app.post("/feedback/{user_id}")
async def feedback_endpoint(user_id: str, req: FeedbackRequest):
    bot = get_bot(user_id) # Get the specific user's bot
    
    if req.label not in bot.personalizer.mood_labels:
        raise HTTPException(status_code=400, detail=f"Invalid label. Must be one of: {bot.personalizer.mood_labels}")
    
    # Run training in a thread so it doesn't block the API
    await asyncio.to_thread(bot.record_mood_feedback, req.label)
    return {"status": "feedback recorded", "user": user_id, "label": req.label}
# --- End Fix 6 ---

@app.get("/ping")
def ping():
    return {"status": "API is running!"}

# --- Interactive console ---
def chat():
    print("Synthia: Hi, I'm Synthia. I’m here to listen. Type 'exit' to quit.\n")
    print("Synthia: (You can also type 'feedback:stressed', 'feedback:calm', or 'feedback:happy' to help me learn!)\n") 
    
    # --- FIX 1: Create a dedicated bot for the console ---
    console_bot = SynthiaCore(
        user_id="console_user",
        llm=llm,
        llm_lock=llm_lock,  # <-- Pass the lock
        sentiment_analyzer=sentiment_analyzer,
        emotion_analyzer=emotion_analyzer
    )
    # --- End Fix 1 ---
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSynthia: Take care!")
            break
        if user_input.lower() in ["exit", "quit"]:
            print("Synthia: Take care!")
            break
        
        if user_input.lower().startswith("feedback:"):
            try:
                parts = user_input.split(":", 1)
                if len(parts) > 1:
                    label = parts[1].strip().lower()
                
                    if label in ["stressed", "calm", "happy"]:
                        # --- FIX 1: Use console_bot instance ---
                        console_bot.record_mood_feedback(label)
                        print(f"Synthia: Thanks! I've noted that you're feeling {label}.\n")
                    else:
                        print(f"Synthia: Sorry, I only understand feedback:stressed, feedback:calm, or feedback:happy.\n")
                else:
                    print("Synthia: Please provide a label, like 'feedback:stressed'.\n")
            except Exception as e:
                print(f"Synthia: Oops, couldn't log that feedback. (Error: {e})\n")
            continue 
        
        # --- FIX 1: Use console_bot instance ---
        reply, _, _, _ = console_bot.generate_response(user_input) 
        print(f"Synthia: {reply}\n")

# --- FIX 1 (Merge): This is the *only* code that should be in the __name__ block ---
if __name__ == "__main__":
    try:
        chat()
    except Exception as e:
        print(f"Fatal error in interactive mode: {e}")