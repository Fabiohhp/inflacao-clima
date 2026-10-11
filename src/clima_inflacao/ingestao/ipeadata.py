"""Séries do IPEADATA (API OData v4), usadas para os preços da Seab-PR e da Conab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from clima_inflacao.config import caminho_projeto
from clima_inflacao.ingestao.cache import ler_ou_baixar, ler_ou_baixar_json
from clima_inflacao.ingestao.cliente_http import obter_json


def _http(config: dict[str, Any]) -> dict[str, Any]:
    """Seção `http` com o timeout próprio do IPEADATA (a API chega a levar minutos por série)."""
    cfg = config["precos_atacado"]["ipeadata"]
    return {**config["http"], "timeout_segundos": cfg["timeout_segundos"]}


def ler_valores(bruto: pd.DataFrame) -> pd.Series:
    """Converte a resposta de ValoresSerie em série mensal indexada pelo 1º dia do mês.

    VALDATA vem como "AAAA-MM-DDT00:00:00-03:00". Valores nulos ficam NaN (nada é preenchido).
    Datas fora do 1º dia do mês ou repetidas geram erro.
    """
    datas = pd.to_datetime(bruto["VALDATA"].astype(str).str[:10], format="%Y-%m-%d")
    if (datas.dt.day != 1).any():
        raise ValueError("série do IPEADATA com datas fora do 1º dia do mês (não é mensal?)")
    if datas.duplicated().any():
        raise ValueError("série do IPEADATA com meses repetidos")
    valores = pd.to_numeric(bruto["VALVALOR"], errors="raise").astype(float)
    return pd.Series(valores.to_numpy(), index=pd.DatetimeIndex(datas, name="mes")).sort_index()


def _baixar(codigo: str, *, config: dict[str, Any], dir_cache: Path, refresh: bool) -> pd.DataFrame:
    url = config["precos_atacado"]["ipeadata"]["url_valores"].format(codigo=codigo)

    def baixar() -> pd.DataFrame:
        resposta = obter_json(url, http=_http(config))
        bruto = pd.DataFrame(resposta["value"], columns=["SERCODIGO", "VALDATA", "VALVALOR"])
        return bruto.astype({"SERCODIGO": str, "VALDATA": str, "VALVALOR": float})

    return ler_ou_baixar(dir_cache / f"valores_{codigo}.parquet", baixar, refresh=refresh)


def metadados(codigo: str, *, config: dict[str, Any], refresh: bool = False) -> dict[str, Any]:
    """Metadados da série (nome, fonte, unidade, comentário), em cache."""
    dir_cache = caminho_projeto("dados_brutos", config) / "ipeadata"
    url = config["precos_atacado"]["ipeadata"]["url_metadados"].format(codigo=codigo)
    resposta = ler_ou_baixar_json(
        dir_cache / f"metadados_{codigo}.json",
        lambda: obter_json(url, http=_http(config)),
        refresh=refresh,
    )
    return resposta["value"][0]


def series_mensais(
    nomes: list[str], *, config: dict[str, Any], refresh: bool = False
) -> pd.DataFrame:
    """Séries de `precos_atacado.series` (pelo nome no settings), uma coluna por série."""
    dir_cache = caminho_projeto("dados_brutos", config) / "ipeadata"
    cfg = config["precos_atacado"]["series"]
    colunas = {}
    for nome in nomes:
        bruto = _baixar(cfg[nome]["codigo"], config=config, dir_cache=dir_cache, refresh=refresh)
        colunas[nome] = ler_valores(bruto)
    return pd.DataFrame(colunas)
