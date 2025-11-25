from fastapi import FastAPI
from pydantic import BaseModel
from modelo.prediccion_modelo import Producto
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import List, Dict
import io
import base64

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
# CONFIGURACIÓN K-MEANS
# ===========================================
# Modelo K-Means
kmeans_model = None
scaler_kmeans = StandardScaler()
df_with_clusters = None

def entrenar_kmeans(k=3):
    """Entrena el modelo K-Means con k clusters"""
    global kmeans_model, df_with_clusters, scaler_kmeans
    
    df_kmeans = pd.read_csv("datos_limpio.csv")
    X_kmeans = df_kmeans[["precio"]]
    
    # Escalar datos
    X_scaled = scaler_kmeans.fit_transform(X_kmeans)
    
    # Entrenar K-Means
    kmeans_model = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans_model.fit_predict(X_scaled)
    
    df_kmeans["cluster"] = clusters
    df_with_clusters = df_kmeans
    
    return kmeans_model

# Entrenar K-Means inicialmente con k=3
entrenar_kmeans(k=3)

# ===========================================
# CONFIGURACIÓN REGRESIÓN LOGÍSTICA
# ===========================================
logistic_model = None
scaler_logistic = StandardScaler()
X_train_log = None
X_test_log = None
y_train_log = None
y_test_log = None
columnas_logistic = []
median_precio = None

def entrenar_regresion_logistica():
    """Entrena modelo de regresión logística para clasificar precios altos/bajos"""
    global logistic_model, scaler_logistic, X_train_log, X_test_log, y_train_log, y_test_log, columnas_logistic, median_precio
    
    df_log = pd.read_csv("datos_limpio.csv")
    
    # Crear variable objetivo (precio alto = 1, bajo = 0)
    median_precio = df_log["precio"].median()
    df_log["precio_alto"] = (df_log["precio"] > median_precio).astype(int)
    
    # Convertir categóricas a dummies
    df_log = pd.get_dummies(df_log, columns=[
        "localidad", "provincia", "region", "producto", "tipohorario"
    ], drop_first=True)
    
    # Separar X e y
    X_log = df_log.drop(columns=["precio", "precio_alto"])
    y_log = df_log["precio_alto"]
    
    columnas_logistic = X_log.columns.tolist()
    
    # Dividir datos
    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
        X_log, y_log, test_size=0.2, random_state=42
    )
    
    # Escalar
    X_train_log = scaler_logistic.fit_transform(X_train_log)
    X_test_log = scaler_logistic.transform(X_test_log)
    
    # Entrenar
    logistic_model = LogisticRegression(max_iter=2000)
    logistic_model.fit(X_train_log, y_train_log)
    
    return logistic_model

# Entrenar regresión logística inicialmente
entrenar_regresion_logistica()

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
        "modelos_disponibles": ["Regresión Lineal", "Regresión Logística", "K-Means Clustering"],
        "endpoints": {
            "regresion_lineal": [
                "/predecir - POST: Predecir precio de un producto",
                "/metricas - GET: Obtener métricas del modelo",
                "/entrenar - POST: Re-entrenar el modelo",
                "/comparacion - GET: Ver comparación real vs predicho"
            ],
            "regresion_logistica": [
                "/logistica/predecir - POST: Clasificar si precio es alto o bajo",
                "/logistica/entrenar - POST: Re-entrenar modelo logístico",
                "/logistica/metricas - GET: Accuracy, precision, recall, F1",
                "/logistica/matriz-confusion - GET: Matriz de confusión",
                "/logistica/reporte - GET: Reporte completo de clasificación"
            ],
            "kmeans": [
                "/kmeans/entrenar - POST: Entrenar K-Means con k clusters",
                "/kmeans/predecir - POST: Predecir cluster de un precio",
                "/kmeans/metodo-codo - GET: Calcular método del codo",
                "/kmeans/clusters - GET: Ver clusters y centroides",
                "/kmeans/estadisticas - GET: Estadísticas por cluster"
            ]
        }
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


# ===========================================
# RUTAS REGRESIÓN LOGÍSTICA
# ===========================================

class ProductoLogistica(BaseModel):
    localidad: str
    provincia: str
    region: str
    producto: str
    tipohorario: str
    latitud: float
    longitud: float
    anio_indice: int
    mes_indice: int
    anio_vigencia: int
    mes_vigencia: int


@app.post("/logistica/predecir")
def predecir_logistica(data: ProductoLogistica):
    """Predice si un producto tendrá precio alto (1) o bajo (0)"""
    if logistic_model is None:
        return {"error": "Modelo logístico no entrenado"}
    
    # Convertir a DataFrame
    df_input = pd.DataFrame([data.dict()])
    
    # Convertir categóricas a dummies
    df_input = pd.get_dummies(df_input, columns=[
        "localidad", "provincia", "region", "producto", "tipohorario"
    ], drop_first=True)
    
    # Asegurar que todas las columnas existan
    for col in columnas_logistic:
        if col not in df_input:
            df_input[col] = 0
    
    # Alinear columnas
    df_input = df_input[columnas_logistic]
    
    # Escalar
    df_input_scaled = scaler_logistic.transform(df_input)
    
    # Predecir
    prediccion = logistic_model.predict(df_input_scaled)[0]
    probabilidad = logistic_model.predict_proba(df_input_scaled)[0]
    
    return {
        "clasificacion": "Precio Alto" if prediccion == 1 else "Precio Bajo",
        "valor_predicho": int(prediccion),
        "probabilidad_bajo": float(probabilidad[0]),
        "probabilidad_alto": float(probabilidad[1]),
        "umbral_mediana": float(median_precio),
        "interpretacion": f"El precio está {'por encima' if prediccion == 1 else 'por debajo'} de la mediana (${median_precio:.2f})"
    }


@app.post("/logistica/entrenar")
def entrenar_logistica_endpoint():
    """Re-entrena el modelo de regresión logística"""
    entrenar_regresion_logistica()
    
    # Calcular accuracy
    y_pred = logistic_model.predict(X_test_log)
    accuracy = accuracy_score(y_test_log, y_pred)
    
    return {
        "mensaje": "Modelo logístico re-entrenado exitosamente",
        "accuracy": float(accuracy),
        "umbral_mediana": float(median_precio)
    }


@app.get("/logistica/metricas")
def metricas_logistica():
    """Retorna métricas del modelo de clasificación"""
    if logistic_model is None:
        return {"error": "Modelo logístico no entrenado"}
    
    y_pred = logistic_model.predict(X_test_log)
    
    return {
        "accuracy": float(accuracy_score(y_test_log, y_pred)),
        "precision": float(precision_score(y_test_log, y_pred)),
        "recall": float(recall_score(y_test_log, y_pred)),
        "f1_score": float(f1_score(y_test_log, y_pred)),
        "umbral_clasificacion": float(median_precio),
        "distribucion_clases": {
            "entrenamiento": {
                "bajos": int((y_train_log == 0).sum()),
                "altos": int((y_train_log == 1).sum())
            },
            "prueba": {
                "bajos": int((y_test_log == 0).sum()),
                "altos": int((y_test_log == 1).sum())
            }
        }
    }


@app.get("/logistica/matriz-confusion")
def matriz_confusion_logistica():
    """Retorna la matriz de confusión del modelo"""
    if logistic_model is None:
        return {"error": "Modelo logístico no entrenado"}
    
    y_pred = logistic_model.predict(X_test_log)
    cm = confusion_matrix(y_test_log, y_pred)
    
    return {
        "matriz_confusion": cm.tolist(),
        "interpretacion": {
            "verdaderos_negativos": int(cm[0][0]),
            "falsos_positivos": int(cm[0][1]),
            "falsos_negativos": int(cm[1][0]),
            "verdaderos_positivos": int(cm[1][1])
        },
        "descripcion": {
            "VN": "Precios bajos correctamente clasificados",
            "FP": "Precios bajos clasificados como altos (error)",
            "FN": "Precios altos clasificados como bajos (error)",
            "VP": "Precios altos correctamente clasificados"
        }
    }


@app.get("/logistica/reporte")
def reporte_completo_logistica():
    """Reporte completo de clasificación"""
    if logistic_model is None:
        return {"error": "Modelo logístico no entrenado"}
    
    y_pred = logistic_model.predict(X_test_log)
    
    # Classification report como diccionario
    report_dict = classification_report(y_test_log, y_pred, output_dict=True)
    
    return {
        "reporte_clasificacion": report_dict,
        "accuracy": float(accuracy_score(y_test_log, y_pred)),
        "total_predicciones": len(y_test_log),
        "correctas": int((y_pred == y_test_log).sum()),
        "incorrectas": int((y_pred != y_test_log).sum()),
        "umbral_precio": float(median_precio)
    }
