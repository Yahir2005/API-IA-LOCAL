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
    model ="dolphin3",
    messages= [
        {"role":"user","content":"¿Qué es la inteligencia artificial?"}
    ]
)

print(response["message"]["content"])

"""
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)

"""

    
