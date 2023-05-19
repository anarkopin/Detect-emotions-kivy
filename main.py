import cv2
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from deepface import DeepFace
import numpy as np
from kivy.graphics.transformation import Matrix
from kivy.graphics.texture import Texture

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


class OpenCVCamera(App):
    def build(self):
        layout = BoxLayout()
        self.capture = cv2.VideoCapture(0)
        self.image = Image()
        

        layout.add_widget(self.image)

        # # Programamos la actualización de la imagen de la cámara
        Clock.schedule_interval(self.update, 10/30.0)
        
        return layout
    
    def update(self, dt):
        # Capturamos el frame de la cámara
        ret, frame = self.capture.read()

        if ret:
            # Procesamos el frame con OpenCV
            processed_frame = self.process_frame(frame)
            
            # Convertimos el frame procesado a una textura de Kivy
            texture = self.convert_frame_to_texture(processed_frame)

            # Mostramos la imagen en la vista de la cámara
            self.image.texture = texture

    x, y, w, h = 100, 100, 200, 200

    def process_frame(self, frame):
        # Girar el frame en sentido horario
      
        # Convertir a escala de grises y detectar rostros en el frame girado
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Dibujar un rectángulo alrededor de cada rostro detectado en el frame original (no girado)
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
        # Convertir la imagen de BGR a RGB (Kivy utiliza el formato RGB)


        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = cv2.flip(frame_rgb, 0)


        return frame
    
    def convert_frame_to_texture(self, frame):
        # Creamos una textura de Kivy a partir del frame procesado
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')

        # Pasamos el contenido del frame a la textura
        texture.blit_buffer(frame.tobytes(), colorfmt='rgb', bufferfmt='ubyte')

        return texture

if __name__ == '__main__':
    OpenCVCamera().run()