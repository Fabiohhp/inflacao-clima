"""Séries do Sistema Gerenciador de Séries Temporais (SGS) do Banco Central."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from clima_inflacao.ingestao.cache import ler_ou_baixar
from clima_inflacao.ingestao.cliente_http import obter_json

URL_SGS = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"


def janelas(
    inicio: pd.Timestamp, fim: pd.Timestamp, anos: int
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Janelas consecutivas de `anos` anos a partir de `inicio`, até cobrir `fim`.

    Os limites são fixos (não dependem da data de hoje), então servem de chave de cache. A
    última janela pode terminar no futuro: a API aceita e devolve até o último dado.
    """
    resultado = []
    atual = inicio
    while atual <= fim:
        proximo = atual + pd.DateOffset(years=anos)
        resultado.append((atual, proximo - pd.Timedelta(days=1)))
        atual = proximo
    return resultado


def _baixar_janela(
    codigo: int,
    inicio: pd.Timestamp,
    fim: pd.Timestamp | None,
    *,
    dir_cache: Path,
    http: dict[str, Any],
    refresh: bool,
) -> pd.DataFrame:
    params = {"formato": "json", "dataInicial": f"{inicio:%d/%m/%Y}"}
    nome = f"sgs_{codigo}_{inicio:%Y%m%d}"
    if fim is not None:
        params["dataFinal"] = f"{fim:%d/%m/%Y}"
        nome += f"_{fim:%Y%m%d}"

    def baixar() -> pd.DataFrame:
        resposta = obter_json(URL_SGS.format(codigo=codigo), params, http=http)
        return pd.DataFrame(resposta, columns=["data", "valor"]).astype(str)

    return ler_ou_baixar(dir_cache / f"{nome}.parquet", baixar, refresh=refresh)


def serie_sgs(
    codigo: int,
    *,
    inicio: pd.Timestamp,
    frequencia: str,
    janela_max_anos: int,
    dir_cache: Path,
    http: dict[str, Any],
    refresh: bool = False,
) -> pd.DataFrame:
    """Série completa a partir de `inicio`, com colunas data e valor.

    Séries diárias são baixadas em janelas de até `janela_max_anos` (limite da API); as demais,
    numa consulta só.
    """
    if frequencia == "diaria":
        limites = janelas(inicio, pd.Timestamp.today().normalize(), janela_max_anos)
    else:
        limites = [(inicio, None)]
    bruto = pd.concat(
        [
            _baixar_janela(codigo, ini, fim, dir_cache=dir_cache, http=http, refresh=refresh)
            for ini, fim in limites
        ],
        ignore_index=True,
    )
    serie = pd.DataFrame(
        {
            "data": pd.to_datetime(bruto["data"], format="%d/%m/%Y"),
            "valor": pd.to_numeric(bruto["valor"], errors="raise"),
        }
    )
    if serie["data"].duplicated().any():
        raise ValueError(f"série {codigo}: datas repetidas entre janelas")
    return serie.sort_values("data").reset_index(drop=True)


def media_mensal(diaria: pd.DataFrame) -> pd.DataFrame:
    """Média mensal de uma série diária, com o número de dias usados (`n_dias`)."""
    mes = diaria["data"].dt.to_period("M").dt.to_timestamp().rename("mes")
    return diaria.groupby(mes)["valor"].agg(valor="mean", n_dias="count").reset_index()
