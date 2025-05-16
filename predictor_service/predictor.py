# predictor.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from text_analyzer import analyze_text
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

class TextRequest(BaseModel):
    text: str

@app.post("/analyze")
def analyze(req: TextRequest):
    return analyze_text(req.text)
