import sqlite3
import pickle
from deepface import DeepFace

# funcion parra conectar o crear base de datos
def crear_conectar_db(nombre_db):
    try:
        # connecta o crea la base de datos en caso de no existir
        conn = sqlite3.connect(nombre_db)
        # activa el cursor
        cursor = conn.cursor()

        # crea la tabla con el nombre y el embedding de la persona en caso de que no exista
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usuarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        ''')
        
        # guarda los cambios de la base de datos
        conn.commit()
        print(f"Creando base de datos: {nombre_db}")
        # retorna la conexion
        return conn
    # en caso de error con la base de datos este lo muestra en la terminal
    except sqlite3.Error as e:
        print(f"Error: {e}")

def agregando_nuevo_rostro(ruta_db, nombre, img):
    conn = crear_conectar_db(ruta_db)
    cursor = conn.cursor()

    try:
        embedding_obj = DeepFace.represent(
            img_path=img,
            model_name='ArcFace',
            detector_backend='opencv'
        )
        embedding = embedding_obj[0]['embedding']
        print(len(embedding))
        emmbeding_blob = pickle.dumps(embedding)

        sql = "INSERT INTO usuarios (nombre, embedding) VALUES (?, ?)"

        cursor.execute(sql, (nombre, emmbeding_blob))    
        conn.commit()
        conn.close()
        print(f"Se agrego el usuario: {nombre}")
    except Exception as e:
        print(f"Error: {e}")

agregando_nuevo_rostro('/home/aidam/Desktop/facial_recognition_spoofing/images/aidam.jpg')
