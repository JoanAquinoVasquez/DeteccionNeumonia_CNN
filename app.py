from flask import Flask, request, render_template, redirect
from tensorflow.keras.models import load_model
import numpy as np
import os
import glob
from PIL import Image

app = Flask(__name__)

# Cargar el modelo
model = load_model('model/Neumonia_CNN.keras')

# Definir las clases
class_names = ['Normal', 'Neumonía']

# Función para limpiar imágenes temporales
def clean_temp_images():
    temp_files = glob.glob('static/temp*.jpg')
    for file in temp_files:
        os.remove(file)

# Función para preprocesar la imagen
def preprocess_image(image):
    image = image.convert('L')  # Convertir a escala de grises
    image = image.resize((500, 500))  # Redimensionar
    image_array = np.array(image) / 255.0  # Normalizar
    image_array = np.expand_dims(image_array, axis=-1)  # Añadir dimensión de canal
    image_array = np.expand_dims(image_array, axis=0)   # Añadir dimensión de batch
    return image_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Limpiar imágenes temporales antes de guardar una nueva
            clean_temp_images()
            
            # Guardar la imagen en un archivo temporal
            file_path = os.path.join('static', 'temp.jpg')
            file.save(file_path)
            
            try:
                # Leer la imagen usando PIL
                image = Image.open(file.stream)
                print(f"Original Image Size: {image.size}")
                
                # Preprocesar la imagen
                preprocessed_image = preprocess_image(image)
                print(f"Preprocessed Image Shape: {preprocessed_image.shape}")
                
                # Hacer la predicción
                prediction = model.predict(preprocessed_image)
                probability = prediction[0][0]  # Obtener la probabilidad de neumonía
                print(f"Prediction Probability: {probability}")
                
                # Determinar la clase con base en la probabilidad
                if probability > 0.5:
                    result = "NEUMONÍA"
                else:
                    result = "NORMAL"
                
                # Calcular los porcentajes
                percentages = [
                    ('Normal', (1 - probability) * 100),
                    ('Neumonía', probability * 100)
                ]
                
                print(f"Prediction: {result}")
                print(f"Probabilities: {percentages}")
                
            except Exception as e:
                print(f"Error during processing or prediction: {e}")
                return render_template('index.html', result=None, image_path=None, percentages=None)
            
            return render_template('index.html', result=result, image_path='temp.jpg', percentages=percentages)
        
    return render_template('index.html', result=None, image_path=None, percentages=None)

if __name__ == '__main__':
    app.run()
