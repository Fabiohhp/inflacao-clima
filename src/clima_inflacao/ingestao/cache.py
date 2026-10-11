"""Cache em disco das respostas brutas das APIs (data/raw).

Reexecutar não baixa de novo: o arquivo em cache é lido, salvo com `refresh=True`.
A gravação é atômica (arquivo temporário + rename), então uma falha no meio do download
não deixa cache corrompido.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

log = logging.getLogger(__name__)


def _gravar_atomico(caminho: Path, escrever: Callable[[Path], None]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_name(caminho.name + ".tmp")
    escrever(temporario)
    temporario.replace(caminho)


def ler_ou_baixar(
    caminho: Path, baixar: Callable[[], pd.DataFrame], *, refresh: bool = False
) -> pd.DataFrame:
    """Lê o Parquet em cache; se não existir (ou com refresh), chama `baixar` e grava."""
    if caminho.exists() and not refresh:
        log.info("cache: %s", caminho.name)
        return pd.read_parquet(caminho)
    df = baixar()
    _gravar_atomico(caminho, lambda destino: df.to_parquet(destino, index=False))
    log.info("gravado: %s (%d linhas)", caminho.name, len(df))
    return df


def ler_ou_baixar_json(caminho: Path, baixar: Callable[[], Any], *, refresh: bool = False) -> Any:
    """Igual a `ler_ou_baixar`, para respostas JSON que não são tabelas (ex.: metadados)."""
    if caminho.exists() and not refresh:
        log.info("cache: %s", caminho.name)
        return json.loads(caminho.read_text(encoding="utf-8"))
    conteudo = baixar()
    _gravar_atomico(
        caminho,
        lambda destino: destino.write_text(json.dumps(conteudo, ensure_ascii=False), "utf-8"),
    )
    log.info("gravado: %s", caminho.name)
    return conteudo


def ler_ou_baixar_texto(caminho: Path, baixar: Callable[[], str], *, refresh: bool = False) -> str:
    """Igual a `ler_ou_baixar`, para respostas em texto puro (ex.: arquivos de largura fixa)."""
    if caminho.exists() and not refresh:
        log.info("cache: %s", caminho.name)
        return caminho.read_text(encoding="utf-8")
    conteudo = baixar()
    _gravar_atomico(caminho, lambda destino: destino.write_text(conteudo, "utf-8"))
    log.info("gravado: %s", caminho.name)
    return conteudo


def ler_ou_baixar_geo(
    caminho: Path, baixar: Callable[[], gpd.GeoDataFrame], *, refresh: bool = False
) -> gpd.GeoDataFrame:
    """Igual a `ler_ou_baixar`, para malhas geográficas (GeoParquet)."""
    if caminho.exists() and not refresh:
        log.info("cache: %s", caminho.name)
        return gpd.read_parquet(caminho)
    gdf = baixar()
    _gravar_atomico(caminho, lambda destino: gdf.to_parquet(destino, index=False))
    log.info("gravado: %s (%d linhas)", caminho.name, len(gdf))
    return gdf
