"""Minimal API to test FastAPI."""
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Test API")

@app.get("/")
def root():
    return {"message": "FastAPI works!", "status": "ok"}

if __name__ == "__main__":
    print("🚀 Starting FastAPI...")
    uvicorn.run(app, host="0.0.0.0", port=8000)


