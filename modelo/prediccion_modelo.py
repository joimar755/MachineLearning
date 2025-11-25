from pydantic import BaseModel

class Producto(BaseModel):
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