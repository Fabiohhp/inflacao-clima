"""Malhas territoriais do IBGE (via geobr), com cache em GeoParquet, e pontos de referência."""

from __future__ import annotations

from typing import Any

import geobr
import geopandas as gpd
import pandas as pd

from clima_inflacao.config import caminho_projeto
from clima_inflacao.ingestao.cache import ler_ou_baixar_geo
from clima_inflacao.ingestao.cliente_http import executar

# Projeção policônica do Brasil (SIRGAS 2000), em metros: o ponto representativo de cada
# polígono é calculado nela e convertido de volta para latitude/longitude.
CRS_METRICO = 5880
CRS_GEOGRAFICO = 4326


def microrregioes(config: dict[str, Any], *, refresh: bool = False) -> gpd.GeoDataFrame:
    """As 558 microrregiões geográficas (código de 5 dígitos, igual ao nível N9 do SIDRA)."""
    ano = config["espacial"]["malhas"]["ano_microrregioes"]
    malha = ler_ou_baixar_geo(
        caminho_projeto("dados_brutos", config) / "geobr" / f"microrregioes_{ano}.parquet",
        lambda: executar(
            lambda: geobr.read_micro_region(year=ano, simplified=True),
            descricao=f"geobr.read_micro_region(year={ano}, simplified=True)",
            http=config["http"],
        ),
        refresh=refresh,
    )
    malha["code_micro"] = malha["code_micro"].astype(int)
    return malha


def sedes_municipais(config: dict[str, Any], *, refresh: bool = False) -> gpd.GeoDataFrame:
    """Pontos das sedes de todos os municípios."""
    ano = config["espacial"]["malhas"]["ano_sedes"]
    sedes = ler_ou_baixar_geo(
        caminho_projeto("dados_brutos", config) / "geobr" / f"sedes_municipais_{ano}.parquet",
        lambda: executar(
            lambda: geobr.read_municipal_seat(year=ano),
            descricao=f"geobr.read_municipal_seat(year={ano})",
            http=config["http"],
        ),
        refresh=refresh,
    )
    sedes["code_muni"] = sedes["code_muni"].astype(int)
    return sedes


def pontos_microrregioes(malha: gpd.GeoDataFrame) -> pd.DataFrame:
    """Um ponto por microrregião, garantidamente dentro do polígono (representative_point)."""
    pontos = malha.to_crs(CRS_METRICO).representative_point().to_crs(CRS_GEOGRAFICO)
    return pd.DataFrame(
        {
            "ponto_id": "micro_" + malha["code_micro"].astype(str),
            "tipo": "microrregiao",
            "codigo": malha["code_micro"].astype(str),
            "nome": malha["name_micro"],
            "uf": malha["abbrev_state"],
            "lat": pontos.y.to_numpy(),
            "lon": pontos.x.to_numpy(),
        }
    )


def pontos_areas_ipca(config: dict[str, Any], sedes: gpd.GeoDataFrame) -> pd.DataFrame:
    """Sede municipal de cada área do IPCA, com a distância até a coordenada do settings.

    Levanta ValueError se algum código de sede do settings não existir na malha.
    """
    areas = pd.DataFrame(config["ipca"]["areas"]["lista"]).dropna(subset=["municipio_sede"])
    areas["municipio_sede"] = areas["municipio_sede"].astype(int)
    pontos = areas.merge(sedes, left_on="municipio_sede", right_on="code_muni", how="left")
    if pontos["name_muni"].isna().any():
        faltam = pontos.loc[pontos["name_muni"].isna(), "id"].tolist()
        raise ValueError(f"sedes sem correspondência no geobr: {faltam}")
    geometria = gpd.GeoSeries(pontos["geometry"], crs=sedes.crs).to_crs(CRS_GEOGRAFICO)
    preliminar = gpd.GeoSeries(gpd.points_from_xy(pontos["lon"], pontos["lat"]), crs=CRS_GEOGRAFICO)
    distancia = geometria.to_crs(CRS_METRICO).distance(preliminar.to_crs(CRS_METRICO)) / 1000
    return pd.DataFrame(
        {
            "ponto_id": "area_" + pontos["id"],
            "tipo": "area_ipca",
            "codigo": pontos["municipio_sede"].astype(str),
            "nome": pontos["name_muni"],
            "uf": pontos["abbrev_state"],
            "lat": geometria.y.to_numpy(),
            "lon": geometria.x.to_numpy(),
            "dist_coord_settings_km": distancia.round(1).to_numpy(),
        }
    )
