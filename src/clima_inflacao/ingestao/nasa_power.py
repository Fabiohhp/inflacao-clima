"""Clima diário do NASA POWER, uma requisição por célula da grade MERRA-2.

Pontos na mesma célula (0,5° x 0,625°) recebem séries idênticas da API, então cada célula é
baixada uma vez, pelo centro, e cada ponto é associado à sua célula.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from clima_inflacao.config import caminho_projeto
from clima_inflacao.ingestao.cache import ler_ou_baixar, ler_ou_baixar_json
from clima_inflacao.ingestao.cliente_http import obter_json

log = logging.getLogger(__name__)


def celulas(pontos: pd.DataFrame, passo_lat: float, passo_lon: float) -> pd.DataFrame:
    """Acrescenta a célula da grade de cada ponto: centro (lat/lon_celula) e `celula_id`."""
    resultado = pontos.copy()
    # "+ 0.0" troca -0.0 por 0.0, para a célula do equador ter um só identificador.
    resultado["lat_celula"] = np.round(np.round(pontos["lat"] / passo_lat) * passo_lat, 4) + 0.0
    resultado["lon_celula"] = np.round(np.round(pontos["lon"] / passo_lon) * passo_lon, 4) + 0.0
    resultado["celula_id"] = [
        f"{lat:.2f}_{lon:.3f}"
        for lat, lon in zip(resultado["lat_celula"], resultado["lon_celula"], strict=True)
    ]
    return resultado


def ler_resposta(resposta: dict[str, Any], valor_ausente: float) -> pd.DataFrame:
    """Converte o JSON da API em tabela bruta (data AAAAMMDD + um parâmetro por coluna).

    Confere o código de ausente declarado no cabeçalho; os valores -999 são mantidos (bruto).
    """
    declarado = resposta["header"]["fill_value"]
    if not np.isclose(declarado, valor_ausente):
        raise ValueError(f"código de ausente {declarado}, esperado {valor_ausente}")
    tabela = pd.DataFrame(resposta["properties"]["parameter"])
    return tabela.rename_axis("data").reset_index()


def baixar_celula(
    lat: float,
    lon: float,
    celula_id: str,
    *,
    config: dict[str, Any],
    fim: pd.Timestamp,
    refresh: bool = False,
) -> pd.DataFrame:
    """Série diária bruta de uma célula, do início do clima até `fim` (cache por célula)."""
    cfg = config["clima"]["nasa_power"]
    inicio = pd.Period(config["periodos"]["clima_inicio"], "M").start_time
    params = {
        "parameters": ",".join(cfg["parametros"].values()),
        "community": cfg["comunidade"],
        "latitude": lat,
        "longitude": lon,
        "start": f"{inicio:%Y%m%d}",
        "end": f"{fim:%Y%m%d}",
        "format": "JSON",
        "time-standard": cfg["padrao_tempo"],
    }
    return ler_ou_baixar(
        caminho_projeto("dados_brutos", config) / "nasa_power" / f"celula_{celula_id}.parquet",
        lambda: ler_resposta(
            obter_json(cfg["url"], params, http=config["http"]), cfg["valor_ausente"]
        ),
        refresh=refresh,
    )


def baixar_clima(
    pontos: pd.DataFrame, config: dict[str, Any], *, refresh: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Baixa todas as células dos pontos e devolve (clima diário, pontos com a célula).

    O clima diário tem uma linha por célula × dia, com os parâmetros renomeados conforme o
    settings e -999 convertido em NaN.
    """
    cfg = config["clima"]["nasa_power"]
    pontos = celulas(pontos, cfg["grade_graus"]["lat"], cfg["grade_graus"]["lon"])
    unicas = pontos.drop_duplicates("celula_id")[["celula_id", "lat_celula", "lon_celula"]]
    fim = pd.Timestamp.today().normalize()
    renomear = {codigo: nome for nome, codigo in cfg["parametros"].items()}
    partes = []
    for i, (celula_id, lat, lon) in enumerate(unicas.itertuples(index=False), start=1):
        bruto = baixar_celula(lat, lon, celula_id, config=config, fim=fim, refresh=refresh)
        dados = bruto.rename(columns=renomear)
        valores = dados[list(renomear.values())].astype("float32")
        valores = valores.mask(np.isclose(valores, cfg["valor_ausente"]))
        partes.append(
            pd.concat(
                [
                    pd.DataFrame(
                        {
                            "celula_id": celula_id,
                            "data": pd.to_datetime(dados["data"], format="%Y%m%d"),
                        }
                    ),
                    valores,
                ],
                axis=1,
            )
        )
        if i % 50 == 0 or i == len(unicas):
            log.info("NASA POWER: %d de %d células", i, len(unicas))
    return pd.concat(partes, ignore_index=True), pontos


def elevacao(
    celulas_unicas: pd.DataFrame, config: dict[str, Any], *, refresh: bool = False
) -> pd.DataFrame:
    """Altitude (m) da grade em cada célula, lida da geometria de uma consulta de um dia.

    Usada pelo ET0 de Penman-Monteith (pressão atmosférica e radiação de céu claro).
    """
    cfg = config["clima"]["nasa_power"]
    dir_cache = caminho_projeto("dados_brutos", config) / "nasa_power" / "elevacao"
    linhas = []
    for celula_id, lat, lon in celulas_unicas[["celula_id", "lat_celula", "lon_celula"]].itertuples(
        index=False
    ):
        params = {
            "parameters": "T2M",
            "community": cfg["comunidade"],
            "latitude": lat,
            "longitude": lon,
            "start": "20200101",
            "end": "20200101",
            "format": "JSON",
        }
        resposta = ler_ou_baixar_json(
            dir_cache / f"celula_{celula_id}.json",
            lambda p=params: obter_json(cfg["url"], p, http=config["http"]),
            refresh=refresh,
        )
        linhas.append(
            {"celula_id": celula_id, "elevacao_m": resposta["geometry"]["coordinates"][2]}
        )
    return pd.DataFrame(linhas)
