from fastapi.testclient import TestClient
from ollama_app import app, encode_image_to_base64

client = TestClient(app)

def test_task_question_with_image(question: str, image_path: str):
    print("--- PROBANDO VISIÓN E IMAGEN ---")
    # Codificamos la imagen
    encoded_image = encode_image_to_base64(image_path)
    
    # Creamos el payload asegurándonos de que dice "encoded_image"
    payload = {
        "question": question,
        "encoded_image": encoded_image 
    }
    
    response = client.post("/question", json=payload)
    assert response.status_code == 200, f"Error: {response.status_code}"
    print("Respuesta Imagen JSON:", response.json())
    print("-" * 30)

def test_busqueda_talento(requerimiento: str):
    print("--- PROBANDO BÚSQUEDA DE TALENTO VECTORIAL ---")
    payload = {
        "requerimiento": requerimiento,
        "limite": 3
    }
    
    response = client.post("/empresas/buscar-talento", json=payload)
    assert response.status_code == 200, f"Error: {response.status_code}"
    print("Respuesta Búsqueda JSON:", response.json())
    print("-" * 30)

if __name__ == "__main__":
    # 1. Prueba de Imagen
    try:
        test_task_question_with_image("¿Qué contiene esta imagen?", "xp.jpeg")
    except Exception as e:
        print("Error en prueba de imagen. ¿Existe xp.jpeg?:", e)
        
    # 2. Prueba de Búsqueda Vectorial 
    # (Nota: Para que esto devuelva algo útil, primero debes haber ejecutado el endpoint 
    # /postulantes/{id}/vectorizar para al menos un postulante)
    try:
        test_busqueda_talento("Busco un desarrollador web que sepa de bases de datos relacionales")
    except Exception as e:
        print("Error en prueba de talento:", e)