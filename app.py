from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from flask_cors import CORS
import os
import requests

# URLs de los archivos en Google Drive
FIREBASE_CREDENTIALS_URL = "https://drive.google.com/uc?id=1Ogevx94z4uCyA-QwIbr__7ym16gy9Exw"
PIPELINE_URL = "https://drive.google.com/uc?id=1ME6it2alIsV-kvC47HJYzY3eEnXJAuzp"
MODEL_URL = "https://drive.google.com/uc?id=1HfAlgRogdOLnCnHHmoAqbaFotHuQu4Cg"

# Rutas donde se guardarán los archivos descargados
FIREBASE_CREDENTIALS_PATH = "firebase-key.json"
PIPELINE_PATH = "models/pipePreprocesadores.pickle"
MODEL_PATH = "models/modeloRF.pickle"

# Función para descargar archivos desde Google Drive
def download_file(url, destination):
    response = requests.get(url)
    if response.status_code == 200:
        with open(destination, "wb") as file:
            file.write(response.content)
        print(f"✅ Archivo descargado correctamente: {destination}")
    else:
        print(f"❌ Error al descargar {destination}, código: {response.status_code}")

# Crear la carpeta models si no existe
os.makedirs("models", exist_ok=True)

# Descargar los archivos necesarios
download_file(FIREBASE_CREDENTIALS_URL, FIREBASE_CREDENTIALS_PATH)
download_file(PIPELINE_URL, PIPELINE_PATH)
download_file(MODEL_URL, MODEL_PATH)

# Inicializar la aplicación Flask
app = Flask(__name__)
CORS(app)  # Habilita CORS para todas las rutas

# Configurar Firebase
cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()
print("✅ Firebase inicializado correctamente.")

# Cargar el modelo y el pipeline desde los archivos descargados
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)
with open(PIPELINE_PATH, "rb") as pipeline_file:
    pipeline = pickle.load(pipeline_file)

print("✅ Modelo y pipeline cargados correctamente.")

@app.route('/')
def home():
    return "✅ API para la predicción con modelo Random Forest funcionando correctamente."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No se enviaron datos."}), 400
        
        # Transformar los datos usando el pipeline
        input_data = pd.DataFrame([data])
        transformed_data = pipeline.transform(input_data)
        
        # Hacer la predicción con el modelo
        prediction = model.predict(transformed_data)
        prediction_proba = model.predict_proba(transformed_data)
        
        # Guardar en Firebase
        prediction_result = {
            "input_data": data,
            "prediction": int(prediction[0]),
            "probability": prediction_proba[0].tolist()
        }
        doc_ref = db.collection("predictions").add(prediction_result)
        prediction_id = doc_ref[1].id

        return jsonify({
            "prediction": prediction_result["prediction"],
            "probability": prediction_result["probability"],
            "predictionId": prediction_id
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    try:
        predictions_ref = db.collection("predictions")
        docs = predictions_ref.stream()
        history = [{"id": doc.id, **doc.to_dict()} for doc in docs]
        return jsonify({"history": history}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/history/<id>', methods=['GET'])
def get_prediction_by_id(id):
    try:
        prediction_ref = db.collection("predictions").document(id)
        doc = prediction_ref.get()
        if doc.exists:
            prediction = doc.to_dict()
            prediction["id"] = doc.id
            return jsonify(prediction), 200
        else:
            return jsonify({"error": f"No se encontró una predicción con ID: {id}"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
