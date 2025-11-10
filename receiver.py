from fastapi import FastAPI, Request
import httpx

app = FastAPI()

@app.post("/receive")
async def receive_data(req: Request):
    data = await req.json()
    print(f"[RECEIVED FROM SYNTHIA BACKEND] {data}")
    
    # ✅ Forward to main webapp (update with your real ngrok/webapp URL)
    webapp_url = "https://intercommunicative-squashier-terrence.ngrok-free.dev/incoming"
    
    try:
        async with httpx.AsyncClient() as client:
            await client.post(webapp_url, json=data)
        print(f"[INFO] Forwarded to WebApp: {webapp_url}")
    except Exception as e:
        print(f"[ERROR] Failed to forward to webapp: {e}")

    return {"status": "ok"}