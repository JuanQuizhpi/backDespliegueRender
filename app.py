from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from flask_cors import CORS
import os
import requests
import json

# URLs de los archivos en Google Drive
PIPELINE_URL = "https://drive.google.com/uc?id=1ME6it2alIsV-kvC47HJYzY3eEnXJAuzp"
MODEL_URL = "https://drive.google.com/uc?id=1HfAlgRogdOLnCnHHmoAqbaFotHuQu4Cg"

# Rutas donde se guardarán los archivos descargados
PIPELINE_PATH = "/tmp/pipePreprocesadores.pickle"
MODEL_PATH = "/tmp/modeloRF.pickle"

# Función para descargar archivos desde Google Drive
def download_file(url, destination):
    response = requests.get(url)
    if response.status_code == 200:
        with open(destination, "wb") as file:
            file.write(response.content)
        print(f"✅ Archivo descargado correctamente: {destination}")
    else:
        print(f"❌ Error al descargar {destination}, código: {response.status_code}")

# Descargar modelos en cada arranque
download_file(PIPELINE_URL, PIPELINE_PATH)
download_file(MODEL_URL, MODEL_PATH)

# Inicializar la aplicación Flask
app = Flask(__name__)
CORS(app)  # Habilita CORS para todas las rutas

# Configurar Firebase usando variables de entorno
firebase_credentials_json = os.environ.get("FIREBASE_CREDENTIALS")
if firebase_credentials_json:
    firebase_credentials = json.loads(firebase_credentials_json)
    cred = credentials.Certificate(firebase_credentials)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase inicializado correctamente.")
else:
    print("❌ No se encontró la variable de entorno FIREBASE_CREDENTIALS")

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
