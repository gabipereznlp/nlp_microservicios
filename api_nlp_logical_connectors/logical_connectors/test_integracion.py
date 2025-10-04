import requests

def test_servicio_en_vivo():
    url = "http://localhost:8000/invertir_texto/"
    params = {"texto": "Hola"}
    response = requests.get(url, params=params)
    assert response.status_code == 200
    assert response.json() == {"Respuesta": "aloH"}
