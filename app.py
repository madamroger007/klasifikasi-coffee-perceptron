from flask import Flask, render_template, request, jsonify
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
import numpy as np
import requests
import dotenv
import pickle  # Untuk menyimpan model

# Memuat file .env untuk API Key
dotenv.load_dotenv()
API_KEY = dotenv.get_key(".env", "API_MEME_KEY")

app = Flask(__name__)

# Data pelatihan untuk berbagai jenis kopi (Arabika, Robusta, Liberica, Excelsa)
X_train = np.array([
    # Arabika
    [7.5, 8.0, 2.0],  # Arabika
    [8.0, 8.5, 3.0],  # Arabika
    [7.8, 7.5, 2.8],  # Arabika
    [7.2, 7.9, 2.2],  # Arabika
    [8.2, 8.0, 3.1],  # Arabika
    [7.6, 7.7, 2.5],  # Arabika
    [7.9, 8.1, 2.7],  # Arabika
    [8.1, 8.3, 2.9],  # Arabika

    # Robusta
    [5.0, 5.5, 6.5],  # Robusta
    [4.5, 5.0, 6.0],  # Robusta
    [6.0, 6.5, 5.5],  # Robusta
    [5.2, 5.8, 6.3],  # Robusta
    [4.7, 5.2, 6.1],  # Robusta
    [5.4, 5.6, 6.2],  # Robusta
    [5.5, 5.9, 6.4],  # Robusta
    [4.8, 5.3, 5.9],  # Robusta

    # Liberica
    [6.8, 6.5, 4.0],  # Liberica
    [6.5, 6.0, 3.9],  # Liberica
    [7.0, 6.7, 4.2],  # Liberica
    [6.7, 6.3, 4.1],  # Liberica
    [6.6, 6.8, 4.3],  # Liberica
    [6.9, 6.9, 4.4],  # Liberica
    [7.2, 6.5, 4.5],  # Liberica
    [6.4, 6.1, 3.8],  # Liberica

    # Excelsa
    [6.0, 6.5, 5.0],  # Excelsa
    [6.5, 6.8, 4.8],  # Excelsa
    [6.2, 6.3, 5.1],  # Excelsa
    [6.7, 7.0, 5.3],  # Excelsa
    [6.3, 6.2, 5.4],  # Excelsa
    [6.8, 6.9, 5.2],  # Excelsa
    [7.0, 6.5, 5.0],  # Excelsa
    [6.5, 6.6, 5.5],  # Excelsa
])

# Label untuk masing-masing jenis kopi (0: Arabika, 1: Robusta, 2: Liberica, 3: Excelsa)
y_train = np.array([
    0, 0, 0, 0, 0, 0, 0, 0,  # Arabika
    1, 1, 1, 1, 1, 1, 1, 1,  # Robusta
    2, 2, 2, 2, 2, 2, 2, 2,  # Liberica
    3, 3, 3, 3, 3, 3, 3, 3   # Excelsa
])

# Label mapping
label_map = {
    0: "Arabika",
    1: "Robusta",
    2: "Liberica",
    3: "Excelsa"
}

# Inisialisasi model Perceptron dan scaler global
scaler = StandardScaler()
model = Perceptron(max_iter=1000, tol=1e-3)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    # Normalisasi data pelatihan
    X_train_scaled = scaler.fit_transform(X_train)

    # Melatih model Perceptron
    model.fit(X_train_scaled, y_train)

    # Simpan model yang telah dilatih
    with open('perceptron_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return jsonify({'message': 'Model berhasil dilatih dengan data kopi!'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    acidity = float(data.get('acidity'))
    aroma = float(data.get('aroma'))
    bitterness = float(data.get('bitterness'))

    # Validasi input
    if not (1.0 <= acidity <= 10.0):
        return jsonify({'error': 'Keasaman harus antara 1.0 dan 10.0.'}), 400
    if not (1.0 <= aroma <= 10.0):
        return jsonify({'error': 'Aroma harus antara 1.0 dan 10.0.'}), 400
    if not (1.0 <= bitterness <= 10.0):
        return jsonify({'error': 'Kepahitan harus antara 1.0 dan 10.0.'}), 400

    # Melakukan prediksi
    input_data = np.array([acidity, aroma, bitterness]).reshape(1, -1)

    # Normalisasi input menggunakan scaler yang sama
    input_data_scaled = scaler.transform(input_data)

    # Memuat model yang sudah dilatih
    with open('perceptron_model.pkl', 'rb') as f:
        model = pickle.load(f)

    prediction = model.predict(input_data_scaled)[0]
    print(prediction)

    # Mengambil gambar meme dari API eksternal
    url = "https://api.apileague.com/retrieve-random-meme?keywords=rocket"
    headers = {'x-api-key': API_KEY}

    response = requests.get(url, headers=headers)
    meme_data = response.json()
    meme_url = meme_data['url']

    # Menyusun hasil prediksi
    result = label_map.get(prediction, "Jenis kopi tidak dikenal")
    
    return jsonify({'prediction': result, 'image_url': meme_url})

if __name__ == '__main__':
    app.run(debug=True)
