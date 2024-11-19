from flask import Flask, request, render_template
import cv2
import numpy as np
import dlib
import os
import base64

app = Flask(__name__)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

@app.route('/', methods=['GET', 'POST'])
def index():
    img_data = None
    if request.method == 'POST':
        action = request.form.get('action')  # Captura la acción seleccionada
        img_data = request.form.get('img_data')  # Recibe la imagen procesada previamente

        if img_data:
            # Decodificar imagen de base64
            image = cv2.imdecode(np.frombuffer(base64.b64decode(img_data), np.uint8), cv2.IMREAD_COLOR)
        elif 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return "No se seleccionó un archivo", 400
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        else:
            return "No se encontró la imagen", 400

        if action == 'detect_landmarks':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                landmarks = predictor(gray, face)
                puntos = [
                    (landmarks.part(17).x, landmarks.part(17).y),
                    (landmarks.part(21).x, landmarks.part(21).y),
                    (landmarks.part(22).x, landmarks.part(22).y),
                    (landmarks.part(26).x, landmarks.part(26).y),
                    (landmarks.part(30).x, landmarks.part(30).y),
                    (landmarks.part(36).x, landmarks.part(36).y),
                    (landmarks.part(38).x, landmarks.part(38).y),
                    (landmarks.part(39).x, landmarks.part(39).y),
                    (landmarks.part(42).x, landmarks.part(42).y),
                    (landmarks.part(43).x, landmarks.part(43).y),
                    (landmarks.part(45).x, landmarks.part(45).y),
                    (landmarks.part(51).x, landmarks.part(51).y),
                    (landmarks.part(57).x, landmarks.part(57).y),
                    (landmarks.part(60).x, landmarks.part(60).y),
                    (landmarks.part(64).x, landmarks.part(64).y),
                ]

                for punto in puntos:
                    size = 2
                    cv2.line(image, (punto[0] - size, punto[1] - size), (punto[0] + size, punto[1] + size), (0, 0, 255), 2)
                    cv2.line(image, (punto[0] - size, punto[1] + size), (punto[0] + size, punto[1] - size), (0, 0, 255), 2)

        elif action == 'flip_horizontal':
            image = cv2.flip(image, 1)  # Invierte la imagen horizontalmente

        elif action == 'increase_brightness':
            image = cv2.convertScaleAbs(image, alpha=1.2, beta=30)  # Aumenta el brillo

        elif action == 'rotate_180':
            image = cv2.rotate(image, cv2.ROTATE_180)  # Gira la imagen 180 grados

        # Codificar imagen a base64 para mantenerla en la vista
        _, buffer = cv2.imencode('.jpg', image)
        img_data = base64.b64encode(buffer).decode('utf-8')

    return render_template('index.html', img_data=img_data)

if __name__ == '__main__':
    app.run(debug=True)
