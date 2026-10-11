"""Relatório de cobertura das séries: início, fim, nº de observações e % de faltantes."""

from __future__ import annotations

import pandas as pd


def cobertura(
    df: pd.DataFrame,
    chaves: list[str],
    *,
    coluna_data: str = "mes",
    coluna_valor: str = "valor",
    freq: str = "MS",
) -> pd.DataFrame:
    """Uma linha por série (combinação de `chaves`).

    início e fim são a primeira e a última data com valor não ausente; `n_esperado` é o número
    de períodos (na frequência `freq`) entre elas e `pct_faltantes` é a fração desses períodos
    sem valor. Séries sem nenhum valor aparecem com n_obs = 0 e 100% de faltantes.
    """
    validos = df.dropna(subset=[coluna_valor])
    resumo = (
        validos.groupby(chaves)[coluna_data]
        .agg(inicio="min", fim="max", n_obs="nunique")
        .reset_index()
    )
    todas = df[chaves].drop_duplicates()
    resumo = todas.merge(resumo, on=chaves, how="left")
    resumo["n_obs"] = resumo["n_obs"].fillna(0).astype(int)
    resumo["n_esperado"] = [
        len(pd.date_range(inicio, fim, freq=freq)) if pd.notna(inicio) else 0
        for inicio, fim in zip(resumo["inicio"], resumo["fim"], strict=True)
    ]
    faltantes = 1 - resumo["n_obs"] / resumo["n_esperado"].where(resumo["n_esperado"] > 0)
    resumo["pct_faltantes"] = (100 * faltantes.fillna(1)).round(2)
    return resumo.sort_values(chaves).reset_index(drop=True)
