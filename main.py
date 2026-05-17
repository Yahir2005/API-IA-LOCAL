from fastapi import FastAPI
from ollama import chat
from pydantic import BaseModel
import uvicorn

class user(BaseModel):
    id:int
    name: str
    email: str
    active: bool = True


response = chat( 
    model ="dolphin-phi",
    messages= [
        {"role":"user","content":"¿Qué es la inteligencia artificial?"}
    ]
)

print(response["message"]["content"])

