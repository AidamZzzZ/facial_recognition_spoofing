import os
import sys
import cv2
import time
import threading
import traceback
import numpy as np
from deepface import DeepFace

# Permite ejecutar el archivo directamente con python src/threads/threads_model.py
if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.db.db import buscar_por_imagen

bounding_box = None
face_box = None
recognized_name = None
recognition_color = (128, 128, 128)
is_processing = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def buscar_por_imagen_frame(frame):
    global face_box, recognized_name, recognition_color, is_processing

    try:
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Frame inválido")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        analysis = DeepFace.find(
            img_path=frame, 
            model_name="retinaFace", 
            enforce_detection=False, 
            align=True
        )
        analysis = analysis if isinstance(analysis, list) else [analysis]
        rostro = analysis[0]

        region = rostro.get('facial_area') or rostro.get('region')
        if region:
            x = int(region.get('x', 0))
            y = int(region.get('y', 0))
            w = int(region.get('w', 0))
            h = int(region.get('h', 0))
            face_box = (x, y, w, h)
            face_crop_bgr = frame[y:y+h, x:x+w]
        else:
            face_crop_bgr = frame
            nombre = buscar_por_imagen(face_crop_bgr, enforce_detection=False)
            if nombre:
                recognized_name = nombre
                recognition_color = (0, 255, 0)
            else:
                recognized_name = 'DESCONOCIDO'
                recognition_color = (128, 128, 128)

    except Exception as e:
        if 'Face could not be detected' in str(e) or 'Face could not be detected' in repr(e):
            face_box = None
            recognized_name = 'DESCONOCIDO'
            recognition_color = (128, 128, 128)
        else:
            print(f"Error procesando frame: {e}")
            traceback.print_exc()
    finally:
        is_processing = False
      
def video_camara(src = 0, funcion = None):
    global bounding_box, is_processing 
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"No se pudo abrir la cámara con índice {src}. Verifica /dev/video* y los permisos de tu usuario.")
        cap.release()
        return

    p_time = 0

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if not is_processing:
            is_processing = True
            threading.Thread(target=funcion, args=(frame.copy(),)).start()
    
        if face_box is not None:
            x, y, w, h = face_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), recognition_color, 2)
            cv2.rectangle(frame, (x, y - 35), (x + w, y), recognition_color, cv2.FILLED)
            cv2.putText(frame, recognized_name, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("deteccion en vivo", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    th1 = threading.Thread(target=video_camara, args=(1, buscar_por_imagen_frame,))
    th1.start()
    th1.join()
