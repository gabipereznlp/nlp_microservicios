from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import os
from spacy.matcher import Matcher
import spacy

# Cargamos el modelo de spaCy
nlp = spacy.load("es_core_news_sm")

app = FastAPI(
    title="Detección de oraciones impersonales",
    description="Una oración impersonal es aquella que no tiene un sujeto agente explícito o cuyo predicado no se refiere a una persona concreta. Este servicio detecta si la frase que se recibe como parámetro es una oración impersonal.",
    version="1.0.0"
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


matcher = Matcher(nlp.vocab)

#definicion de matchers faltan definir mas matchers para poder eliminar las listas de palabras y que sea independiente de ellas
pattern_opinion = [
    {"POS": "VERB"},      # un verbo
    {"LOWER": "que"}      # seguido de "que" ej: pienso que
]
matcher.add("OPINION", [pattern_opinion])

pattern_percepcion = [
    {"POS": "VERB"},     
    {"POS": "DET", "OP": "?"},  # puede o no ir un determinante (la, el, un...)
    {"POS": "NOUN"}      # y después un sustantivo  ej: veo la pelicula
]
matcher.add("PERCEPCION", [pattern_percepcion])

PALABRAS_OPINION = {
    "gustar", "encantar", "amar", "querer", "disfrutar", "apreciar", "preferir",
    "odiar", "detestar", "rechazar", "desagradar", "fastidiar", "abominar",
    "creer", "pensar", "considerar", "imaginar", "suponer", "asumir",
    "evaluar", "valorar", "criticar", "elogiar", "aprobar", "desaprobar",
    "sentir", "temer", "esperar", "doler", "sufrir", "alegrar", "emocionar",
    "parecer", "opinar", "juzgar", "reflexionar",
    "admirar", "respetar", "confiar", "dudar", "sospechar", "intuir",
}

PALABRAS_PERCEPCION = {
    "ver", "mirar", "observar", "notar", "distinguir", "percibir", "sentir", "escuchar",
    "oler", "tocar", "saborear", "examinar", "inspeccionar", "explorar", "investigar",
    "detectar", "identificar", "reconocer", "descubrir", "advertir", "captar",
    "visualizar", "contemplar", "fijarse", "atender",
    "analizar", "interpretar", "comprender", "entender", "apreciar"
}

@app.get("/opinion-percepcion/")
def opinion_percepcion(texto: str):
    doc = nlp(texto)
    resultado = []
    detectados = set()

    # detectar por patrones con Matcher
    for match_id, start, end in matcher(doc):
        span = doc[start:end]
        label = nlp.vocab.strings[match_id].lower()  # "opinion" o "percepcion"
        lema = span.root.lemma_.lower()

        resultado.append({
            "verbo": span.root.text,   # verbo principal de la frase
            "tipo": label,
            "lema": lema
        })
        detectados.add(lema)  # evitar duplicados

    # detectar por listas de palabras
    for token in doc:
        if token.pos_ == "VERB":
            tipo = []
            lema = token.lemma_.lower()
            if lema not in detectados:  # evitar duplicados
                detectados.add(lema)
                if lema in PALABRAS_OPINION:
                    tipo.append("opinion")
                if lema in PALABRAS_PERCEPCION:
                    tipo.append("percepcion")

                if tipo:
                    resultado.append({
                        "verbo": token.text,
                        "tipo": "-".join(tipo),
                        "lema": lema
                    })

    return {"resultado": resultado}