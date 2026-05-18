from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import base64
import ollama 
import logging
import chromadb
from vector_db import generar_vector_postulante, buscar_talento_vectorial

# --- Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(filename='response.log', level=logging.INFO)

def log_response(logger: logging.Logger, response_data):
    logger.info(f"Log: {response_data}")

# --- Modelos Pydantic ---
class QABase(BaseModel):
    question: str
    answer: str

class QAAnalytics(QABase):
    thought: str
    topic: str

class QuestionPayload(BaseModel):
    question: str
    encoded_image: str

class ChatRequest(BaseModel):
    message: str

# --- Funciones Auxiliares ---
def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# --- Aplicación FastAPI ---
app = FastAPI(title="Futurework IA API")

# Habilitar CORS para que PHP/JS pueda comunicarse con FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# SEGURO: get_or_create_collection evita que la API truene si la DB vectorial está vacía al inicio
chroma_client = chromadb.PersistentClient(path="./candidatos_vdb")
coleccion = chroma_client.get_or_create_collection(name="postulantes")


# Endpoint 1: Pregunta sobre una imagen
@app.post("/question")
def llm_qa_response(payload: QuestionPayload):
    try:
        response = ollama.chat(
            model='dolphin3', 
            messages=[
                {'role': 'system', 'content': 'Eres un ayudante útil.'},
                {'role': 'user', 'content': payload.question, 'images': [payload.encoded_image]}
            ],
            format=QAAnalytics.model_json_schema()
        )
        qa_instance = QAAnalytics.model_validate_json(response.message.content)
        log_response(logger, qa_instance.model_dump())
        return qa_instance
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint 2: Vectorizar a un Postulante
@app.post("/postulantes/{id_postulante}/vectorizar")
def vectorizar_postulante(id_postulante: int):
    exito = generar_vector_postulante(id_postulante)
    if not exito:
        raise HTTPException(status_code=404, detail="Postulante no encontrado o error en la base de datos.")
    return {"mensaje": "Perfil vectorizado correctamente."}


# Endpoint 3: El motor del Chatbot RAG
@app.post("/chat/reclutador")
async def chat_reclutador(request: ChatRequest):
    user_message = request.message # Asegúrate de que aquí diga .message o .mensaje según tu Pydantic
    
    try:
        # CONTROL DE SEGURIDAD: Verificar cuántos candidatos hay indexados
        total_candidatos = coleccion.count()
        
        if total_candidatos == 0:
            # Si no hay nadie vectorizado, le pasamos un texto vacío al prompt del sistema
            candidatos_encontrados = "No hay ningún candidato registrado o vectorizado en la plataforma todavía."
        else:
            # 1. Convertir la consulta del reclutador en un vector
            response_emb = ollama.embeddings(model="nomic-embed-text", prompt=user_message)
            query_embedding = response_emb['embedding']
            
            # Si hay menos de 3 candidatos en total, le pedimos a Chroma sólo los que existan
            limite_resultados = min(3, total_candidatos)
            
            # 2. Buscar en ChromaDB de forma segura
            resultados = coleccion.query(
                query_embeddings=[query_embedding],
                n_results=limite_resultados
            )
            candidatos_encontrados = "\n".join(resultados['documents'][0]) if resultados['documents'] else "No se encontraron candidatos."
        
        # 3. Construir el Prompt del sistema inyectando el contexto obtenido
        system_prompt = f"""
        Eres el asistente inteligente de reclutamiento para la plataforma Futurework ITT.
        Tu objetivo es ayudar a los reclutadores a encontrar candidatos ideales basándote en la información proporcionada a continuación.
        
        CANDIDATOS DISPONIBLES EN LA PLATAFORMA:
        {candidatos_encontrados}
        
        Instrucciones:
        - Si hay candidatos que coincidan con la búsqueda, menciónalos de manera profesional, incluyendo sus nombres, habilidades y correos.
        - Si la lista de candidatos está vacía o dices que no hay nadie registrado, dile amablemente al reclutador que por el momento no hay perfiles dados de alta en el sistema.
        """
        
        # 4. Enviar todo a Ollama (Asegúrate de usar 'dolphin3' si es el que tienes descargado)
        response = ollama.chat(model="dolphin3", messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ])
        
        return {"response": response.message.content}
        
    except Exception as e:
        # ESTO ES CLAVE: Imprime el error real en tu terminal de Linux para saber qué pasó
        print(f"\n[ERROR CRÍTICO EN /chat/reclutador]: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ollama_app:app", reload=True)