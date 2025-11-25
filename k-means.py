import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ------------------------------
# 1. Cargar dataset
# ------------------------------
df = pd.read_csv("datos_limpio.csv")
print("Dataset cargado correctamente:")
print(df.head())

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
# 5. Entrenar K-Means (usando K=3 del método del codo)
# ------------------------------
K = 3
kmeans = KMeans(n_clusters=K, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Agregar columna cluster
df["cluster"] = clusters
print("\nDatos con clusters:")
print(df.head())

# ------------------------------
# 6. Centroides (convertidos a valores originales)
# ------------------------------
centroides = scaler.inverse_transform(kmeans.cluster_centers_)
centroides_df = pd.DataFrame(centroides, columns=["precio"])

print("\nCentroides de cada cluster (en precio real):")
print(centroides_df)

# ------------------------------
# 7. Rangos de precios por cluster
# ------------------------------
print("\nRangos de precio por cluster:")
for c in sorted(df["cluster"].unique()):
    min_p = df[df["cluster"] == c]["precio"].min()
    max_p = df[df["cluster"] == c]["precio"].max()
    print(f"Cluster {c}: {min_p} - {max_p}")

# ------------------------------
# 8. Gráfico final de clusters
# ------------------------------
""" plt.figure(figsize=(8,5))
plt.scatter(df["precio"], df["cluster"], c=df["cluster"], cmap="viridis")
plt.title("Clusters creados usando únicamente el precio")
plt.xlabel("Precio")
plt.ylabel("Cluster asignado")
plt.grid()
plt.show() """

# ------------------------------
# 9. Interpretación automática 
# ------------------------------
print("\nInterpretación automática:")
for i, row in centroides_df.iterrows():
    print(f"Cluster {i}: productos alrededor de un precio medio de {row['precio']:.2f}")
