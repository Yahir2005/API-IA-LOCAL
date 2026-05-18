import pymysql
from vector_db import generar_vector_postulante

# 1. Conexión rápida a tu MariaDB
db = pymysql.connect(
    host="localhost",
    user="admin",            # <-- Pon tu usuario de MariaDB
    password="0110",    # <-- Pon tu contraseña de MariaDB
    database="Futurework_ITT",
    cursorclass=pymysql.cursors.DictCursor
)

try:
    with db.cursor() as cursor:
        # Extraemos todos los IDs de los postulantes que insertaste
        cursor.execute("SELECT idPostulante FROM Postulante")
        postulantes = cursor.fetchall()
        
        print(f"Se encontraron {len(postulantes)} postulantes en MariaDB.")
        print("Iniciando vectorización masiva en ChromaDB...\n")
        
        # 2. Los pasamos uno por uno por tu motor de embeddings
        for p in postulantes:
            id_p = p['idPostulante']
            print(f"-> Sincronizando postulante con ID: {id_p}...")
            generar_vector_postulante(id_p)
            
    print("\n[ÉXITO] ¡Todos los perfiles han sido vectorizados y guardados en ChromaDB!")

except Exception as e:
    print(f"Ocurrió un error durante la sincronización: {e}")

finally:
    db.close()