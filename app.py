from flask import Flask, render_template, request, jsonify
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
import numpy as np
from data_train import X_train, y_train
import requests
import dotenv
import pickle  # Untuk menyimpan model

# Memuat file .env untuk API Key
dotenv.load_dotenv()
API_KEY = dotenv.get_key(".env", "API_MEME_KEY")

app = Flask(__name__)

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
