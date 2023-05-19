import cv2
from deepface import DeepFace
import numpy as np

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError('cannot open webcam')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
        if len(result) > 0 and 'dominant_emotion' in result[0]:
            emotion = result[0]['dominant_emotion']

            if emotion == 'angry':
                emotion_text = 'Enfadado'
            elif emotion == 'disgust':
                emotion_text = 'Asqueado'
            elif emotion == 'fear':
                emotion_text = 'Asustado'
            elif emotion == 'happy':
                emotion_text = 'Contento'
            elif emotion == 'sad':
                emotion_text = 'Triste'
            elif emotion == 'surprise':
                emotion_text = 'Sorprendido'
            else:
                emotion_text = 'Neutral'

            font = cv2.FONT_HERSHEY_SIMPLEX
            # Posición del texto justo encima del recuadro de la cara
            text_x = x
            text_y = y - 10

            cv2.putText(frame,
                        emotion_text,
                        (text_x, text_y),
                        font, 1,
                        (0, 0, 255),
                        2,
                        cv2.LINE_4)

    cv2.imshow('Original video', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
