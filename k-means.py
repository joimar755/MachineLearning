import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ------------------------------
# 1. Cargar dataset
# ------------------------------
df = pd.read_csv("datos_limpio.csv")

# ------------------------------
# 2. Seleccionar solo la columna precio
# ------------------------------
X = df[["precio"]]

# ------------------------------
# 3. Escalar los datos (muy importante para K-Means)
# ------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# 4. Método del codo para elegir K
# ------------------------------
inertia_values = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia_values.append(km.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K_range, inertia_values, marker="o")
plt.title("Método del Codo para Seleccionar K (solo precio)")
plt.xlabel("Número de clusters (K)")
plt.ylabel("Inercia")
plt.grid()
plt.show()

# ------------------------------
# 5. Entrenar K-Means (ejemplo con K=3)
# ------------------------------
K = 3
kmeans = KMeans(n_clusters=K, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Agregar columna cluster
df["cluster"] = clusters

print(df.head())

# ------------------------------
# 6. Visualización simple
# ------------------------------
plt.figure(figsize=(8,5))
plt.scatter(df["precio"], df["cluster"], c=df["cluster"])
plt.title("Clusters creados usando únicamente precio")
plt.xlabel("Precio")
plt.ylabel("Cluster asignado")
plt.grid()
plt.show()
