from fastapi.middleware.cors import CORSMiddleware
import unicodedata
from fastapi import FastAPI
from fastapi.params import Query
from pydantic import BaseModel
from collections import Counter
import spacy
from spacy.tokens import Token
from fastapi import Request
from fastapi.responses import JSONResponse


# Cargamos el modelo de spaCy
nlp = spacy.load("es_core_news_sm")

# Modelo de entrada
app = FastAPI(title="Detección de repetición de palabras", version="1.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class TextoEntrada(BaseModel):
    texto: str


def _es_palabra_frecuente(token: Token) -> bool:
    # si es articulo
    if "Art" in token.morph.get("PronType"):
        return True
    # si es preposicion
    if token.pos_ == "ADP":
        return True
    if token.pos_ == "PRON":
        return True
    # si es conjuncion coordinante
    if token.pos_ == "CCONJ":
        return True
    # si es conjuncion subordinante
    if token.pos_ == "SCONJ":
        return True
    return False


def _normalizar_token(
        token: Token,
        sin_palabras_frecuentes: bool,
        con_sustantivos_en_singular: bool) -> str | None:
    # ignora signos de puntuacion
    if token.is_punct:
        return None
    # ignora palabras frecuentes
    if sin_palabras_frecuentes and _es_palabra_frecuente(token):
        return None
    # convierte sustantivos plurales a singular
    if con_sustantivos_en_singular:
        if token.pos_ == "NOUN" and "Plur" in token.morph.get("Number"):
            return token.lemma_.lower()
    # conserva nombres propios en mayusculas
    if token.pos_ == "PROPN":
        return token.text
    return token.text.lower()


def _clave_alfabetica_sin_tildes(palabra: str) -> str:
    base = unicodedata.normalize("NFD", palabra)
    sin_tildes = "".join(ch for ch in base if not unicodedata.combining(ch))
    return sin_tildes.lower()

def _contar_palabras_repetidas(palabras: list[str]) -> dict[str, int]:
    contador_de_palabras = Counter(palabras)

    resultado = {
        palabra: cantidad_de_apariciones
        for palabra, cantidad_de_apariciones in contador_de_palabras.items()
        if cantidad_de_apariciones > 1
    }

    resultado_ordenado_descendente_por_cantidad_de_repeticiones = {
        palabra: cantidad_apariciones
        for palabra, cantidad_apariciones in sorted(
            resultado.items(), key=lambda item: (-item[1], _clave_alfabetica_sin_tildes(item[0]), item[0])

            #resultado.items(), key=lambda item: item[1], reverse=True
        )
    }

    return resultado_ordenado_descendente_por_cantidad_de_repeticiones


# Endpoint principal: POST /repeticiones
@app.post("/repeticiones")
async def detectar(
    entrada: TextoEntrada,
    sin_palabras_frecuentes: bool = Query(
        False, description="Ignorar artículos, pronombres, preposiciones y conjunciones"
    ),
    con_sustantivos_en_singular: bool = Query(
        False, description="Llevar sustantivos plurales a singular"
    )
):
    doc = nlp(entrada.texto)
    tokens = [token for token in doc]

    palabras = [
        palabra
        for token in tokens
        if (palabra := _normalizar_token(token, sin_palabras_frecuentes, con_sustantivos_en_singular))
           is not None
    ]

    return _contar_palabras_repetidas(palabras)

# Endpoint de prueba
@app.get("/")
def root():
    return {"mensaje": "API Detector de repeticiones de palabras. Usa POST /repeticiones con JSON { 'texto': '...' }"}

