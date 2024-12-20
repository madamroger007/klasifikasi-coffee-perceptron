import numpy as np
from sklearn.preprocessing import StandardScaler

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        """
        Inisialisasi model Perceptron.
        :param learning_rate: Kecepatan pembelajaran.
        :param epochs: Jumlah iterasi pelatihan.
        """
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None  # Bobot diinisialisasi dengan None
        self.bias = None     # Bias diinisialisasi dengan None
        self.classes = []    # Daftar kelas yang digunakan untuk prediksi

    def fit(self, X, y):
        """
        Melatih model Perceptron menggunakan data latih.
        :param X: Data fitur (numpy array), shape (n_samples, n_features).
        :param y: Label target (numpy array), shape (n_samples,).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, len(set(y))))  # Bobot untuk setiap kelas
        self.bias = np.zeros(len(set(y)))  # Bias untuk setiap kelas
        self.classes = list(set(y))  # Menyimpan kelas yang ada dalam data

        # Menormalisasi data menggunakan StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X_scaled):
                # Hitung output untuk setiap kelas
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i[:, None]
                self.bias += update

    def activation_function(self, X):
        """
        Fungsi aktivasi softmax untuk multikelas.
        :param X: Nilai keluaran linear untuk setiap kelas.
        :return: Probabilitas untuk setiap kelas.
        """
        exp_values = np.exp(X - np.max(X))  # Menghindari overflow
        return exp_values / exp_values.sum(axis=0)

    def predict(self, X):
        """
        Melakukan prediksi untuk input yang diberikan.
        :param X: Data fitur (numpy array), shape (n_samples, n_features).
        :return: Prediksi kelas (numpy array).
        """
        # Menormalisasi data menggunakan StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        linear_output = np.dot(X_scaled, self.weights) + self.bias
        probabilities = self.activation_function(linear_output)
        return np.argmax(probabilities, axis=1)  # Mengembalikan kelas dengan probabilitas tertinggi
