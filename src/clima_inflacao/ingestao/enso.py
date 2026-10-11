"""Índices ENSO da NOAA PSL (ONI e MEI v2), em arquivos texto de largura fixa."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from clima_inflacao.config import caminho_projeto
from clima_inflacao.ingestao.cache import ler_ou_baixar_texto
from clima_inflacao.ingestao.cliente_http import obter


def ler_psl(texto: str) -> tuple[pd.DataFrame, float]:
    """Lê o formato da NOAA PSL sem depender do tamanho do rodapé.

    Formato: 1ª linha com anos inicial e final; linhas "ano + 12 valores"; depois, uma linha só
    com o código de ausente; depois, texto livre. Devolve (série mensal, código de ausente);
    valores iguais ao código viram NaN.
    """
    linhas = texto.splitlines()
    ano_inicial, ano_final = (int(campo) for campo in linhas[0].split()[:2])
    registros: list[list[str]] = []
    ausente = None
    for linha in linhas[1:]:
        campos = linha.split()
        if len(campos) == 13 and campos[0].isdigit():
            registros.append(campos)
        elif registros and len(campos) == 1:
            ausente = float(campos[0])
            break
    anos = [int(r[0]) for r in registros]
    if anos != list(range(ano_inicial, ano_final + 1)):
        raise ValueError(f"anos inesperados no arquivo: {anos[:3]}...{anos[-3:]}")
    if ausente is None:
        raise ValueError("linha com o código de ausente não encontrada")

    largo = pd.DataFrame(
        [[float(v) for v in r[1:]] for r in registros],
        index=pd.Index(anos, name="year"),
        columns=pd.Index(range(1, 13), name="month"),
    )
    serie = largo.stack().rename("valor").reset_index().assign(day=1)
    serie["mes"] = pd.to_datetime(serie[["year", "month", "day"]])
    serie["valor"] = serie["valor"].mask(np.isclose(serie["valor"], ausente))
    return serie[["mes", "valor"]], ausente


def baixar_enso(config: dict[str, Any], *, refresh: bool = False) -> pd.DataFrame:
    """ONI e MEI v2 em formato longo (mes, indice, valor); confere o código de ausente."""
    dir_cache = caminho_projeto("dados_brutos", config) / "noaa"
    partes = []
    for nome, cfg in config["clima"]["enso"].items():
        texto = ler_ou_baixar_texto(
            dir_cache / f"{nome}.txt",
            lambda url=cfg["url"]: obter(url, http=config["http"]).text,
            refresh=refresh,
        )
        serie, ausente = ler_psl(texto)
        if not np.isclose(ausente, cfg["valor_ausente"]):
            raise ValueError(
                f"{nome}: código de ausente {ausente}, esperado {cfg['valor_ausente']}"
            )
        partes.append(serie.assign(indice=nome))
    enso = pd.concat(partes, ignore_index=True)
    return enso[["mes", "indice", "valor"]]
