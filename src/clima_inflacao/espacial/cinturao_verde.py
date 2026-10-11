"""Cinturão verde: clima agregado num raio em torno da sede de cada área do IPCA."""

from __future__ import annotations

import pandas as pd

from clima_inflacao.ingestao.inmet import distancia_km


def celulas_no_raio(
    sedes: pd.DataFrame, celulas: pd.DataFrame, raios_km: list[float]
) -> pd.DataFrame:
    """Células cujo centro fica a até cada raio da sede.

    `sedes` tem area_id, lat e lon; `celulas` tem celula_id, lat_celula e lon_celula. Cada
    célula aparece uma vez por área e raio.
    """
    linhas = []
    for sede in sedes.itertuples(index=False):
        distancias = distancia_km(sede.lat, sede.lon, celulas["lat_celula"], celulas["lon_celula"])
        for raio in raios_km:
            dentro = celulas.assign(distancia_km=distancias)[distancias <= raio]
            linhas.append(
                dentro[["celula_id", "distancia_km"]].assign(area_id=sede.area_id, raio_km=raio)
            )
    return pd.concat(linhas, ignore_index=True)


def agregar(features: pd.DataFrame, selecao: pd.DataFrame, variaveis: list[str]) -> pd.DataFrame:
    """Média simples das células selecionadas, por área, raio e mês, com o nº de células."""
    dados = selecao.merge(features[["celula_id", "mes", *variaveis]], on="celula_id")
    agregado = dados.groupby(["area_id", "raio_km", "mes"])[variaveis].mean()
    agregado["n_celulas"] = dados.groupby(["area_id", "raio_km", "mes"])["celula_id"].nunique()
    return agregado.reset_index()
