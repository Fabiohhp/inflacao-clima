"""Variáveis diárias derivadas (ET0, GDD, extremos) e agregação diária -> mensal por célula."""

from __future__ import annotations

from typing import Any

import pandas as pd

from clima_inflacao.features import balanco_hidrico, gdd
from clima_inflacao.features.extremos import maior_sequencia


def nome_gdd(t_base: float, t_teto: float) -> str:
    return f"gdd_{t_base:g}_{t_teto:g}"


def diarias_derivadas(
    diario: pd.DataFrame, celulas: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Acrescenta ET0 (duas versões), GDD, graus-dia de calor e as marcas de dia de geada/calor.

    `celulas` traz celula_id, lat_celula e elevacao_m; `cfg` é a seção `features` do settings.
    As marcas de geada e calor ficam NaN quando a temperatura do dia falta.
    """
    dados = diario.merge(
        celulas[["celula_id", "lat_celula", "elevacao_m"]], on="celula_id", how="left",
        validate="m:1",
    )  # fmt: skip
    dia_do_ano = dados["data"].dt.dayofyear
    tmax, tmin = dados["temp_max_c"], dados["temp_min_c"]
    dados["et0_pm_mm"] = balanco_hidrico.et0_penman_monteith(
        tmax, tmin, dados["umidade_rel_pct"], dados["vento_2m_ms"], dados["radiacao_mj_m2"],
        dados["lat_celula"], dia_do_ano, dados["elevacao_m"],
        albedo=cfg["balanco_hidrico"]["albedo"],
    )  # fmt: skip
    dados["et0_hg_mm"] = balanco_hidrico.et0_hargreaves(tmax, tmin, dados["lat_celula"], dia_do_ano)
    for limites in {(c["t_base_c"], c["t_teto_c"]) for c in cfg["gdd"].values()}:
        dados[nome_gdd(*limites)] = gdd.graus_dia(tmax, tmin, *limites)
    dados["graus_dia_calor"] = gdd.graus_dia_calor(tmax, cfg["graus_dia_calor"]["limiar_c"])
    dados["dia_geada"] = (tmin <= cfg["geada"]["tmin_limite_c"]).astype(float).where(tmin.notna())
    dados["dia_calor"] = (
        (tmax > cfg["calor_extremo"]["tmax_limite_c"]).astype(float).where(tmax.notna())
    )
    return dados.drop(columns=["lat_celula", "elevacao_m"])


def mensal(dados: pd.DataFrame, *, exigir_mes_completo: bool = True) -> pd.DataFrame:
    """Agrega o diário derivado em meses, por célula.

    Chuva, ET0 e graus-dia são somados; Tmax e Tmin, médias (tmed = média das duas); geada e
    calor, contagens de dias, mais a maior sequência de calor no mês e a dummy de geada. Cada
    variável só recebe valor se todos os dias do mês tiverem dado (com `exigir_mes_completo`).
    """
    dados = dados.sort_values(["celula_id", "data"])
    mes = dados["data"].dt.to_period("M").dt.to_timestamp().rename("mes")
    grupos = [dados["celula_id"], mes]
    somas = ["precipitacao_mm", "et0_pm_mm", "et0_hg_mm", "graus_dia_calor"]
    somas += [c for c in dados.columns if c.startswith("gdd_")]
    medias = {"temp_max_c": "tmax_media_c", "temp_min_c": "tmin_media_c"}
    contagens = {"dia_geada": "dias_geada", "dia_calor": "dias_calor"}
    agregados = {c: (c, "sum") for c in somas}
    agregados |= {novo: (c, "mean") for c, novo in medias.items()}
    agregados |= {novo: (c, "sum") for c, novo in contagens.items()}
    validos = {c: (c, "count") for c in [*somas, *medias, *contagens]}
    agregado = dados.groupby(grupos).agg(**agregados)
    n_validos = dados.groupby(grupos).agg(**validos)
    agregado["max_seq_calor"] = maior_sequencia(dados["dia_calor"], grupos)

    if exigir_mes_completo:
        dias_mes = agregado.index.get_level_values("mes").days_in_month.to_numpy()
        saida_por_coluna_diaria = {c: c for c in somas} | medias | contagens
        for coluna_dia, coluna_saida in saida_por_coluna_diaria.items():
            incompleto = n_validos[coluna_dia].to_numpy() < dias_mes
            agregado.loc[incompleto, coluna_saida] = float("nan")
        agregado.loc[n_validos["dia_calor"].to_numpy() < dias_mes, "max_seq_calor"] = float("nan")

    agregado["tmed_c"] = (agregado["tmax_media_c"] + agregado["tmin_media_c"]) / 2
    agregado["geada"] = (
        (agregado["dias_geada"] >= 1).astype(float).where(agregado["dias_geada"].notna())
    )
    agregado["balanco_pm_mm"] = agregado["precipitacao_mm"] - agregado["et0_pm_mm"]
    agregado["balanco_hg_mm"] = agregado["precipitacao_mm"] - agregado["et0_hg_mm"]
    return agregado.reset_index()
