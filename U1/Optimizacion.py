import tensorflow as tf

# Datos de ejemplo
X = [...]  # características
y = [...]  # etiquetas

# Parámetros del modelo
W = tf.Variable(tf.random.normal(shape=(num_features, 1)), name='weights')
b = tf.Variable(tf.zeros(shape=(1,)), name='bias')

# Hiperparámetros
learning_rate = 0.01
num_epochs = 100
batch_size = 32

# Función de costo
def mean_squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Optimizador
optimizer = tf.optimizers.SGD(learning_rate)

# Entrenamiento del modelo
for epoch in range(num_epochs):
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]

        with tf.GradientTape() as tape:
            y_pred = tf.matmul(X_batch, W) + b
            loss = mean_squared_error(y_batch, y_pred)

        gradients = tape.gradient(loss, [W, b])
        optimizer.apply_gradients(zip(gradients, [W, b]))

    print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')

# Evaluación del modelo
# Evaluar el modelo en un conjunto de datos de prueba