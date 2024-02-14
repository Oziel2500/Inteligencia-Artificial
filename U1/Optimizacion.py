import torch

# Datos de ejemplo (convertidos a tensores de PyTorch)
X = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)  # características
y = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0], dtype=torch.float32)  # etiquetas

# Parámetros del modelo
W = torch.randn((num_features, 1), requires_grad=True)  # Inicialización aleatoria de los pesos
b = torch.zeros(1, requires_grad=True)  # Inicialización de sesgo

# Hiperparámetros
learning_rate = 0.01
num_epochs = 100
batch_size = 32

# Función de costo
def mean_squared_error(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

# Optimizador
optimizer = torch.optim.SGD([W, b], lr=learning_rate)

# Entrenamiento del modelo
for epoch in range(num_epochs):
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]

        # Predicción del modelo
        y_pred = torch.matmul(X_batch, W) + b

        # Cálculo de la función de pérdida
        loss = mean_squared_error(y_batch, y_pred)

        # Retropropagación y actualización de parámetros
        optimizer.zero_grad()  # Reiniciar los gradientes
        loss.backward()  # Retropropagación
        optimizer.step()  # Actualización de parámetros

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Evaluación del modelo
# Evaluar el modelo en un conjunto de datos de prueba