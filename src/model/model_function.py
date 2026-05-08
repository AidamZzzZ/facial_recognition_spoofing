import cv2
import supervision as sv
from ultralytics import YOLO
import face_recognition
from deepface import DeepFace
from deepface.modules.exceptions import ImgNotFound

model = YOLO('yolov8n.pt')

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

def detectar_rostro_imagen(base_path):
    try: 
        img = cv2.imread(base_path)
        result = DeepFace.extract_faces(img_path=base_path, detector_backend='retinaface', enforce_detection=False)

        for face in result:
            confianza = face['confidence']
            if confianza > 0.6: 
                x = face['facial_area']['x']
                y = face['facial_area']['y']
                w = face['facial_area']['w']
                h = face['facial_area']['h']
                
            
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, f"{confianza:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        imagen_redimensionada = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("resultado: ",imagen_redimensionada)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return img
    except cv2.error as e:
        print(f"Error: {e}")
    except ImgNotFound as e:
        print(f"Error: {e}")
        
def detectar_rostro_vivo():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Camara no pudo iniciarse")
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_location = face_recognition.face_locations(rgb_frame, model='arcface')
        
        for (top, right, bottom, left) in face_location:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        cv2.imshow("Deteccion facial en vivo", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()