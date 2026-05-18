import pymysql
import ollama
import chromadb

# Configuración de conexión a MariaDB
def obtener_conexion_db():
    return pymysql.connect(
        host="localhost",
        user="admin",            # <-- Ajusta tu usuario de MariaDB
        password="0110",    # <-- Ajusta tu contraseña
        database="Futurework_ITT",
        cursorclass=pymysql.cursors.DictCursor
    )

def generar_vector_postulante(id_postulante: int) -> bool:
    """
    Consulta los datos de un postulante en MariaDB, genera su string de texto,
    crea el embedding con Ollama y lo guarda/actualiza en ChromaDB.
    """
    try:
        db = obtener_conexion_db()
        with db.cursor() as cursor:
            # 1. Obtener datos base del Postulante
            cursor.execute("SELECT Usuarios_idUsuarios, ubicacion FROM Postulante WHERE idPostulante = %s", (id_postulante,))
            postulante = cursor.fetchone()
            if not postulante:
                print(f"Postulante con ID {id_postulante} no encontrado en la DB.")
                db.close()
                return False
            
            # 2. Obtener datos del Usuario
            cursor.execute("SELECT nombreCompleto, email FROM Usuarios WHERE idUsuarios = %s", (postulante['Usuarios_idUsuarios'],))
            usuario = cursor.fetchone()
            
            # 3. Obtener sus Habilidades
            cursor.execute("""
                SELECT h.nombreHabilidad FROM Postulante_Habilidades ph
                INNER JOIN Habilidades h ON ph.Habilidades_idHabilidad = h.idHabilidad
                WHERE ph.Postulante_idPostulante = %s
            """, (id_postulante,))
            habilidades = [row['nombreHabilidad'] for row in cursor.fetchall()]
            
            # 4. Obtener sus Certificaciones
            cursor.execute("""
                SELECT c.nombre FROM Postulante_Certificacion pc
                INNER JOIN Certificaciones c ON pc.Certificaciones_idCertificacion = c.idCertificacion
                WHERE pc.Postulante_idPostulante = %s
            """, (id_postulante,))
            certificaciones = [row['nombre'] for row in cursor.fetchall()]
        
        db.close()

        # 5. Crear el texto plano estructurado
        texto_perfil = f"Postulante: {usuario['nombreCompleto']}. Email: {usuario['email']}. Ubicación: {postulante['ubicacion']}. "
        texto_perfil += f"Habilidades: {', '.join(habilidades)}. "
        texto_perfil += f"Certificaciones: {', '.join(certificaciones)}."

        # 6. Inicializar ChromaDB y guardar el vector
        chroma_client = chromadb.PersistentClient(path="./candidatos_vdb")
        coleccion = chroma_client.get_or_create_collection(name="postulantes")
        
        # Generar embedding con Ollama
        response_emb = ollama.embeddings(model="nomic-embed-text", prompt=texto_perfil)
        embedding = response_emb['embedding']
        
        # Guardar o actualizar (Upsert)
        coleccion.upsert(
            ids=[str(id_postulante)],
            embeddings=[embedding],
            documents=[texto_perfil],
            metadatas=[{"id_usuario": postulante['Usuarios_idUsuarios'], "nombre": usuario['nombreCompleto']}]
        )
        return True

    except Exception as e:
        print(f"Error al generar vector para el postulante {id_postulante}: {e}")
        return False

def buscar_talento_vectorial(query: str):
    """
    Función auxiliar por si necesitas hacer búsquedas rápidas 
    directamente desde este módulo.
    """
    try:
        chroma_client = chromadb.PersistentClient(path="./candidatos_vdb")
        coleccion = chroma_client.get_or_create_collection(name="postulantes")
        
        response_emb = ollama.embeddings(model="nomic-embed-text", prompt=query)
        query_embedding = response_emb['embedding']
        
        resultados = coleccion.query(query_embeddings=[query_embedding], n_results=3)
        return resultados
    except Exception as e:
        print(f"Error en la búsqueda vectorial: {e}")
        return None