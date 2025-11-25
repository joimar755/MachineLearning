from fastapi import FastAPI
from pydantic import BaseModel
from modelo.prediccion_modelo import Producto
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

app = FastAPI()

# ===========================================
# 1. CARGAR Y ENTRENAR EL MODELO SOLO UNA VEZ
# ===========================================

df = pd.read_csv("datos_limpio.csv")

# Convertir categóricas en dummies
df_dummies = pd.get_dummies(df, drop_first=True)

X = df_dummies.drop(columns=["precio"])
y = df_dummies["precio"]

# Guardamos las columnas que usará el modelo
columnas_modelo = X.columns.tolist()

# División de datos
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrenar modelo
model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)


# ===========================================
# 2. MODELO PARA RECIBIR UN PRODUCTO
# ===========================================



# ===========================================
# 3. RUTA PARA HACER PREDICCIÓN
# ===========================================

@app.get("/")
def root():
    return {
        "mensaje": "API de Machine Learning",
        "endpoints": [
            "/predecir - POST: Predecir precio de un producto",
            "/metricas - GET: Obtener métricas del modelo",
            "/entrenar - POST: Re-entrenar el modelo",
            "/comparacion - GET: Ver comparación real vs predicho"
        ]
    }


@app.post("/predecir")
def predecir(data: Producto):

    # Convertir el JSON a DataFrame
    df_input = pd.DataFrame([data.dict()])

    # Convertir categóricas a dummies igual que con el dataset original
    df_input = pd.get_dummies(df_input)

    # Asegurar que todas las columnas del modelo existan
    for col in columnas_modelo:
        if col not in df_input:
            df_input[col] = 0

    # Alinear columnas exactamente en el mismo orden
    df_input = df_input[columnas_modelo]

    # Predicción
    prediccion = model.predict(df_input)[0]
    
    # Validación: si el precio es negativo, ajustar al mínimo razonable
    if prediccion < 0:
        prediccion = max(prediccion, 0)
        mensaje = "Predicción ajustada a 0 (valor negativo detectado)"
    else:
        mensaje = "Predicción exitosa"

    return {
        "precio_predicho": float(prediccion),
        "r2_modelo": float(r2),
        "mensaje": mensaje,
        "rango_entrenamiento": {
            "min": float(y_train.min()),
            "max": float(y_train.max())
        }
    }


@app.get("/metricas")
def obtener_metricas():
    """Retorna las métricas del modelo entrenado"""
    return {
        "r2_score": float(r2),
        "numero_features": len(columnas_modelo),
        "tamaño_entrenamiento": len(x_train),
        "tamaño_prueba": len(x_test)
    }


@app.post("/entrenar")
def re_entrenar():
    """Re-entrena el modelo con los datos actuales"""
    global model, r2, x_train, x_test, y_train, y_test
    
    df = pd.read_csv("datos_limpio.csv")
    df_dummies = pd.get_dummies(df, drop_first=True)
    
    X = df_dummies.drop(columns=["precio"])
    y = df_dummies["precio"]
    
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    
    return {
        "mensaje": "Modelo re-entrenado exitosamente",
        "nuevo_r2": float(r2)
    }


@app.get("/comparacion")
def obtener_comparacion(limite: int = 10):
    """Retorna comparación entre valores reales y predichos"""
    y_pred = model.predict(x_test)
    
    comparacion = pd.DataFrame({
        "Real": y_test.values[:limite],
        "Predicho": y_pred[:limite]
    })
    
    return {
        "comparacion": comparacion.to_dict(orient="records"),
        "r2_score": float(r2)
    }
