import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("datos_limpio.csv")


# Convertir columnas categóricas en dummies
df = pd.get_dummies(df, drop_first=True)

x = df.drop(columns=["precio"])
y = df["precio"]

# Separar datos
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)

print("Nuevo R²:", r2)
print(y_pred) 

comparacion = pd.DataFrame({
    "Real": y_test.values,
    "Predicho": y_pred
})


print(comparacion.head(5))

""" # Gráfica
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Real")
plt.plot(y_pred, label="Predicho")
plt.title("Comparación Precio Real vs Predicho")
plt.xlabel("Índice")
plt.ylabel("Precio")
plt.legend()
plt.tight_layout()
plt.show() """

