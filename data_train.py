import numpy as np
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
