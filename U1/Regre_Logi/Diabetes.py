import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Cargar datos desde el archivo CSV
data = pd.read_csv('C:/Users/Adan/Documents/IA/Ejercicios/U1/Regre_Logi/diabetes2.csv')

# Dividir los datos en características (X) y etiquetas (y)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características para mejorar el rendimiento del modelo
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar el modelo de regresión logística
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Calcular las probabilidades de pertenecer a la clase 1 (diabetes)
probabilities = model.predict_proba(X_test_scaled)[:, 1]

# Obtener el intercepto
intercepto = model.intercept_[0]
print("Intercepto:", intercepto)

# Obtener los coeficientes de las variables independientes
coeficientes_edad = model.coef_[0]
print("Coeficiente edad:", coeficientes_edad[0])

sexo_coef = model.coef_[0]
print("Coeficiente sexo:", sexo_coef[0])

#datos de un paciente
Edad_Paciente = 38
Sexo_Paciente = 0 #1 Hombre, 0 Mujer

# Calculando la probabilidad individual
z = intercepto + coeficientes_edad[0] * Edad_Paciente + sexo_coef[0] * Sexo_Paciente
probabilidad = 1 / (1 + np.exp(-z))

print("\nLa probabilidad de que una mujer de 38 años pueda padecer diabetes",probabilidad)