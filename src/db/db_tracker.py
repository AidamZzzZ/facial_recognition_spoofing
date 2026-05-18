import os
# Configuración vital para que TensorFlow no arroje Segmentation Fault en hilos
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import cv2
import psycopg2
import threading
import queue
from deepface import DeepFace
from ultralytics import YOLO

def worker_process_face(q, tracked_identities, processing_ids):
    """
    Hilo trabajador en segundo plano. Espera imágenes en la cola, ejecuta DeepFace y 
    consulta la Base de Datos sin bloquear el video.
    """
    try:
        # Abrimos una conexión exclusiva para este hilo
        conn = psycopg2.connect(
            host="localhost",
            user="aidam",
            password="aidam",
            port="5432",
            database="db_faces"
        )
        cursor = conn.cursor()
    except Exception as e:
        print(f"[Worker] Error conectando a BD: {e}")
        return

    while True:
        # Obtenemos la imagen y el ID de la cola (se bloquea hasta recibir algo)
        item = q.get()
        if item is None:
            break # Señal de parada
            
        track_id, face_img = item
        
        # Mensaje en la terminal solicitado
        print(f"\n[Terminal] Procesando ID: {track_id}...")

        try:
            # Extracción del embedding
            # Ya está cargado en caché en la memoria desde el hilo principal
            objs = DeepFace.represent(
                img_path=face_img, 
                model_name="ArcFace", 
                enforce_detection=False,
                align=True
            )
            
            embedding = objs[0].get('embedding') if isinstance(objs, list) and len(objs) > 0 else None
            
            if embedding is not None:
                # Búsqueda en DB
                cursor.execute("""
                SELECT * FROM (
                    SELECT name, embedding <=> %s::vector AS distance
                    FROM users
                ) a
                where distance < 0.6
                ORDER BY distance ASC
                LIMIT 1;
                """, (embedding,))
                
                res = cursor.fetchone()
                if res and res[1] is not None:
                    tracked_identities[track_id] = res[0]
                    print(f"[Terminal] Procesamiento ID: {track_id} terminado. Nombre: {res[0]}")
                else:
                    tracked_identities[track_id] = "Desconocido"
                    print(f"[Terminal] Procesamiento ID: {track_id} terminado. No encontrado en BD.")
            else:
                tracked_identities[track_id] = "Desconocido"
                print(f"[Terminal] Procesamiento ID: {track_id} falló (Sin rostro válido).")
                
        except Exception as e:
            print(f"[Worker] Error procesando rostro ID {track_id}: {e}")
            conn.rollback() # Limpiar error de transacción SQL
            tracked_identities[track_id] = "Desconocido"
            
        finally:
            q.task_done()
            # Quitamos el ID de procesamiento en curso
            processing_ids.discard(track_id)

def main():
    print("Cargando Modelos...")
    
    # Pre-cargamos el modelo ArcFace en el hilo principal
    # ¡Esto evita el Segmentation Fault cuando TF/Keras intenta cargar el modelo en un hilo secundario!
    print("Pre-cargando modelo ArcFace en memoria...")
    DeepFace.build_model("ArcFace")
    
    # Ya estamos utilizando yolov8n.pt (la versión "Nano", que es la más liviana posible para CPU)
    print("Cargando modelo YOLO Nano...")
    model = YOLO("yolo11n.pt")
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    tracked_identities = {}
    processing_ids = set() # Evitar saturar la cola con la misma persona
    
    # Cola para comunicación entre hilos
    q = queue.Queue()
    
    # Iniciar el Worker en segundo plano
    worker_thread = threading.Thread(target=worker_process_face, args=(q, tracked_identities, processing_ids), daemon=True)
    worker_thread.start()

    cap = cv2.VideoCapture(0)
    print("Captura iniciada. Presione 'q' para salir.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()

                for box, track_id in zip(boxes, track_ids):
                    px1, py1, px2, py2 = map(int, box)
                    
                    person_crop = frame[py1:py2, px1:px2]
                    
                    if person_crop.size == 0:
                        continue
                        
                    gray_person = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray_person, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                    
                    if len(faces) > 0:
                        fx, fy, fw, fh = faces[0]
                        face_x1 = px1 + fx
                        face_y1 = py1 + fy
                        face_x2 = face_x1 + fw
                        face_y2 = face_y1 + fh
                        
                        face_img = frame[face_y1:face_y2, face_x1:face_x2]
                        
                        # Si no sabemos quién es y tampoco lo estamos procesando actualmente
                        if track_id not in tracked_identities and track_id not in processing_ids:
                            processing_ids.add(track_id)
                            # Se envía una COPIA del cuadro de la cara a la cola
                            q.put((track_id, face_img.copy()))
                        
                        # Obtener nombre (si todavía no está, será "Desconocido")
                        name = tracked_identities.get(track_id, "Desconocido")
                        
                        color = (150, 150, 150) if name == "Desconocido" else (0, 255, 0)
                            
                        cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), color, 2)
                        cv2.putText(frame, f"ID:{track_id} {name}", (face_x1, face_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    else:
                        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 1)
                        cv2.putText(frame, f"Person ID:{track_id}", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv2.imshow("Tracker y Deteccion de Rostro", frame)

            if cv2.waitKey(1) == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

