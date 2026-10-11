"""Gravação das tabelas processadas em Parquet (data/processed) e no DuckDB (lakehouse)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from clima_inflacao.config import caminho_projeto, carregar_config

log = logging.getLogger(__name__)

_NOME_VALIDO = re.compile(r"^[a-z][a-z0-9_]*$")


def salvar(df: pd.DataFrame, nome: str, config: dict[str, Any] | None = None) -> Path:
    """Grava `df` em data/processed/<nome>.parquet e como tabela <nome> no DuckDB."""
    if not _NOME_VALIDO.match(nome):
        raise ValueError(f"nome de tabela inválido: {nome!r}")
    config = config if config is not None else carregar_config()
    parquet = caminho_projeto("dados_processados", config) / f"{nome}.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet, index=False)
    with duckdb.connect(str(caminho_projeto("lakehouse", config))) as conexao:
        conexao.register("df_temporario", df)
        conexao.execute(f"CREATE OR REPLACE TABLE {nome} AS SELECT * FROM df_temporario")
        conexao.unregister("df_temporario")
    log.info("tabela %s: %d linhas (Parquet + DuckDB)", nome, len(df))
    return parquet


def ler(nome: str, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Lê uma tabela processada a partir do Parquet."""
    config = config if config is not None else carregar_config()
    return pd.read_parquet(caminho_projeto("dados_processados", config) / f"{nome}.parquet")
