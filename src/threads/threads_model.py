import cv2
import time
import threading
import queue
from deepface import DeepFace

bounding_box = None
is_processing = False

def analizar_rostro_bounding_box(frame):
    global bounding_box, is_processing
    results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')

    for result in results:
        bounding_box = (result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h'])        # aplicando downscaling a la imagen a 1/4 parte de su tamano    
    is_processing = False

def analizar_rostro_db(frame):
    global bounding_box, is_processing
    results = DeepFace.find(frame, db_path="/home/aidam/Desktop/facial_recognition_spoofing/src/db", model_name="VGG-Face", enforce_detection=False)
    print(results)
    if not results[0].empty:
        match_path = results[0].iloc[0]['identity']
        person_name = match_path.split("/")[-1].split(".")[0]
        
        cv2.putText(frame, person_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        print("Persona noo existe en la base de dats")
def video_camara(src = 0, funcion = None):
    global bounding_box, is_processing 
    cap = cv2.VideoCapture(src)

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
    
        if bounding_box:
            x, y, w, h = bounding_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("deteccion en vivo", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

#th1 = threading.Thread(target=video_camara, args=(0, analizar_rostro_db,))
#th1.start()
#th1.join()

DeepFace.stream(db_path="src/db", enable_face_analysis=False, model_name="VGG-Face")