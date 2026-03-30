from pydantic import BaseModel,Field
from ollama import chat
import base64


class QABase(BaseModel):
    question: str = Field(...)
    answer: str = Field(..., description="Succinct answer in Spanish Mexico")

class QAAnalytics(QABase)
    thoughts: str = Field(...,description="Thought process that went into answer")
    topic: str =Field(...,description="Single word that best describes the topic of the question")

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_path:
        return base64.b64encode(image_file.read().decode("utf-8"))

def ollama_llm_response(question:str,encode_image:str):
    return chat(
        messages=[
            {
                "role":"system","content":"You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": f"Answer this question: {question}",
                "images": [encode_image]
            },
        ],
        model= "dolphin3",
        format = QAAnalytics.model_json_schema(),
        
    )