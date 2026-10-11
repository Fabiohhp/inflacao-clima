"""Testes do cliente HTTP, do cache e do relatório de cobertura."""

import numpy as np
import pandas as pd
import pytest
import requests

from clima_inflacao.ingestao import cliente_http
from clima_inflacao.ingestao.cache import ler_ou_baixar, ler_ou_baixar_json
from clima_inflacao.ingestao.cobertura import cobertura

HTTP_RAPIDO = {
    "timeout_segundos": 1,
    "tentativas_max": 3,
    "backoff_inicial_segundos": 0,
    "backoff_max_segundos": 0,
}


class RespostaFalsa:
    def __init__(self, status: int):
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


def test_cache_parquet_baixa_uma_vez(tmp_path):
    chamadas = []

    def baixar():
        chamadas.append(1)
        return pd.DataFrame({"a": [1, 2]})

    caminho = tmp_path / "x.parquet"
    primeiro = ler_ou_baixar(caminho, baixar)
    segundo = ler_ou_baixar(caminho, baixar)
    pd.testing.assert_frame_equal(primeiro, segundo)
    assert len(chamadas) == 1
    ler_ou_baixar(caminho, baixar, refresh=True)
    assert len(chamadas) == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_cache_json(tmp_path):
    caminho = tmp_path / "meta.json"
    assert ler_ou_baixar_json(caminho, lambda: {"nome": "Índice"}) == {"nome": "Índice"}
    assert ler_ou_baixar_json(caminho, lambda: pytest.fail("não devia baixar")) == {
        "nome": "Índice"
    }


def test_obter_repete_erro_transitorio(monkeypatch):
    respostas = iter([requests.ConnectionError("caiu"), RespostaFalsa(503), RespostaFalsa(200)])

    def get_falso(*args, **kwargs):
        resposta = next(respostas)
        if isinstance(resposta, Exception):
            raise resposta
        return resposta

    monkeypatch.setattr(cliente_http.requests, "get", get_falso)
    assert cliente_http.obter("http://exemplo", http=HTTP_RAPIDO).status_code == 200


def test_obter_nao_repete_erro_definitivo(monkeypatch):
    chamadas = []

    def get_falso(*args, **kwargs):
        chamadas.append(1)
        return RespostaFalsa(404)

    monkeypatch.setattr(cliente_http.requests, "get", get_falso)
    with pytest.raises(requests.HTTPError):
        cliente_http.obter("http://exemplo", http=HTTP_RAPIDO)
    assert len(chamadas) == 1


def test_cobertura_conta_lacunas_internas():
    meses = pd.date_range("2020-01-01", periods=6, freq="MS")
    df = pd.DataFrame(
        {
            "serie": ["a"] * 6 + ["b"] * 6,
            "mes": list(meses) * 2,
            "valor": [np.nan, 1, 2, np.nan, 4, np.nan] + [np.nan] * 6,
        }
    )
    resumo = cobertura(df, ["serie"]).set_index("serie")
    assert resumo.loc["a", "inicio"] == meses[1]
    assert resumo.loc["a", "fim"] == meses[4]
    assert resumo.loc["a", "n_obs"] == 3
    assert resumo.loc["a", "n_esperado"] == 4
    assert resumo.loc["a", "pct_faltantes"] == 25.0
    assert resumo.loc["b", "n_obs"] == 0
    assert resumo.loc["b", "pct_faltantes"] == 100.0


def test_obter_json_repete_corpo_invalido(monkeypatch):
    class RespostaJson(RespostaFalsa):
        def __init__(self, corpo):
            super().__init__(200)
            self.corpo = corpo

        def json(self):
            if self.corpo is None:
                raise requests.JSONDecodeError("Expecting value", "", 0)
            return self.corpo

    respostas = iter([RespostaJson(None), RespostaJson([{"valor": "1"}])])
    monkeypatch.setattr(cliente_http.requests, "get", lambda *a, **k: next(respostas))
    assert cliente_http.obter_json("http://exemplo", http=HTTP_RAPIDO) == [{"valor": "1"}]
