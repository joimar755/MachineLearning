import pandas as pd
import numpy as np 
df = pd.read_csv('datos_limpio.csv')
print(df)
print(df.info())


print('primeras filas del dataset')
print(df.head())


print('estadisticas descriptivas')
print(df.describe())


if "region" in df.columns:
   df['region']=df['region'].fillna('No especificado')
   
if "geojson" in df.columns:
    df = df.drop(columns=["geojson"])

df = df.dropna(subset=['latitud', 'longitud'])

if "indice_tiempo" in df.columns:
    df['indice_tiempo'] = pd.to_datetime(df['indice_tiempo'], errors='coerce')
    df['anio_indice'] = df['indice_tiempo'].dt.year
    df['mes_indice']  = df['indice_tiempo'].dt.month
else:
    print("⚠️ La columna 'indice_tiempo' no existe en el CSV")



columnas_eliminar = [
    "cuit",
    "empresa",
    "direccion",
    "empresabandera",
    "geojson"
]

df = df.drop(columns=columnas_eliminar, errors='ignore')

print("Columnas después de eliminar:")
print(df.columns)

print('cantidad e filas con valores nulos')
print(df.isnull().sum())

#df = df.drop(columns=["idproducto", "idtipohorario", "idempresa", "idempresabandera"])


#df.to_csv('datos_limpio.csv', index=False)
