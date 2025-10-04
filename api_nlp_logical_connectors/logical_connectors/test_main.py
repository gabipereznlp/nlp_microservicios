import json
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

with open("ejemplos.json", encoding="utf-8") as f:
    casos = json.load(f)

@pytest.mark.parametrize("caso", casos)
def test_invertir_desde_json(caso):
    entrada = caso["input"]
    esperado = caso["output"]
    response = client.get(f"/invertir_texto/?texto={entrada}")
    assert response.status_code == 200
    assert response.json() == {"Respuesta": esperado}
