from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(
    title="Servicio de Inversión de Texto",
    description="Este servicio invierte el texto ingresado. Ideal para pruebas, juegos o manipulación de strings.",
    version="1.0.0"
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/invertir_texto/",
    summary="Invierte el texto proporcionado",
    description="""
Este microservicio recibe un texto y lo devuelve invertido.
La estrategia utilizada es simplemente acceder a los caracteres del texto en orden inverso (`texto[::-1]`).
""",
    response_description="Texto invertido"
)
def invertir_texto(
    texto: str = Query(
        ..., 
        description="Texto que se desea invertir.",
        example="Hola Mundo", 
        
    )
):
    texto_invertido = texto[::-1]
    return {"respuesta": texto_invertido}