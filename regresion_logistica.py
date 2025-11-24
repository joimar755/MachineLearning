import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ================================================================
# 1. Cargar dataset
# ================================================================
df = pd.read_csv("datos_limpio.csv")   # Cambia por tu archivo

# ================================================================
# 2. Crear variable objetivo (clasificación)
# ================================================================
df["precio_alto"] = (df["precio"] > df["precio"].median()).astype(int)

# ================================================================
# 3. Convertir variables categóricas a dummies
# ================================================================
df = pd.get_dummies(df, columns=[
    "localidad", 
    "provincia", 
    "region", 
    "producto", 
    "tipohorario"
], drop_first=True)

# ================================================================
# 4. Separar variables predictoras X y la variable objetivo y
# ================================================================
X = df.drop(columns=["precio", "precio_alto"])
y = df["precio_alto"]

# ================================================================
# 5. Dividir datos en entrenamiento y prueba
# ================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================================================
# 6. Escalar datos numéricos (mejora el modelo)
# ================================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================================================================
# 7. Modelo de regresión logística
# ================================================================
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# ================================================================
# 8. Predicción
# ================================================================
y_pred = model.predict(X_test)

# ================================================================
# 9. Métricas del modelo
# ================================================================
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))
