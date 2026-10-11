"""Validação do SPI contra secas documentadas (critério de aceite da Parte 2)."""

from __future__ import annotations

from typing import Any

import pandas as pd


def spi_microrregioes(spi_mensal: pd.DataFrame, pontos: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """SPI de cada microrregião (pela célula do seu ponto): colunas codigo, uf, mes e `coluna`."""
    micro = pontos[pontos["tipo"] == "microrregiao"][["codigo", "uf", "celula_id"]]
    return micro.merge(spi_mensal[["celula_id", "mes", coluna]], on="celula_id").drop(
        columns="celula_id"
    )


def validar_secas(
    spi_micro: pd.DataFrame, eventos: list[dict[str, Any]], coluna: str, limiar: float = -1.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Para cada evento, a série do SPI médio das microrregiões das UFs e o resumo na janela.

    O evento passa se o SPI médio chega a <= `limiar` em algum mês da janela. Devolve
    (resumo por evento, séries mensais por evento).
    """
    resumos, series = [], []
    for evento in eventos:
        regiao = spi_micro[spi_micro["uf"].isin(evento["ufs"])]
        media = regiao.groupby("mes")[coluna].mean().rename("spi_medio").reset_index()
        series.append(media.assign(evento=evento["evento"]))
        inicio, fim = pd.Timestamp(evento["inicio"]), pd.Timestamp(evento["fim"])
        janela = media[(media["mes"] >= inicio) & (media["mes"] <= fim)].dropna()
        pior = janela.loc[janela["spi_medio"].idxmin()]
        no_pior = regiao[regiao["mes"] == pior["mes"]][coluna]
        resumos.append(
            {
                "evento": evento["evento"],
                "ufs": ", ".join(evento["ufs"]),
                "janela": f"{inicio:%Y-%m} a {fim:%Y-%m}",
                "n_microrregioes": regiao["codigo"].nunique(),
                "spi_medio_minimo": round(pior["spi_medio"], 2),
                "mes_do_minimo": f"{pior['mes']:%Y-%m}",
                "pct_meses_spi_negativo": round(100 * (janela["spi_medio"] < 0).mean(), 1),
                "pct_micro_abaixo_de_1_5_no_minimo": round(100 * (no_pior < -1.5).mean(), 1),
                "passou": bool(pior["spi_medio"] <= limiar),
            }
        )
    return pd.DataFrame(resumos), pd.concat(series, ignore_index=True)


def homogeneidade_nasa_inmet(
    inmet_mensal: pd.DataFrame,
    features: pd.DataFrame,
    estacoes_celulas: pd.DataFrame,
    base: tuple[int, int],
    min_meses_semestre: int = 5,
) -> pd.DataFrame:
    """Viés do NASA POWER contra o INMET por semestre, relativo ao viés de cada estação na base.

    Chuva: razão (NASA / INMET do semestre) / (NASA / INMET da base). Temperatura: diferença
    (tmed NASA − temperatura INMET) menos a diferença média da base. Uma fonte homogênea fica
    perto de 1 e de 0; uma quebra aparece como mudança persistente de patamar. Devolve, por
    semestre, as medianas entre estações e o nº de estações.
    """
    dados = inmet_mensal.merge(estacoes_celulas[["codigo", "celula_id"]], on="codigo").merge(
        features[["celula_id", "mes", "precipitacao_mm", "tmed_c"]],
        on=["celula_id", "mes"],
        suffixes=("_inmet", "_nasa"),
    )
    dados["semestre"] = (
        dados["mes"].dt.year.astype(str) + "-S" + ((dados["mes"].dt.month > 6) + 1).astype(str)
    )
    na_base = dados["mes"].dt.year.between(*base)
    linhas = []
    for variavel in ["chuva", "temperatura"]:
        if variavel == "chuva":
            validos = dados.dropna(subset=["precipitacao_mm_inmet", "precipitacao_mm_nasa"])
            soma = validos.groupby(["codigo", "semestre"]).agg(
                nasa=("precipitacao_mm_nasa", "sum"),
                inmet=("precipitacao_mm_inmet", "sum"),
                meses=("mes", "count"),
            )
            base_est = (
                validos[na_base.loc[validos.index]]
                .groupby("codigo")
                .agg(nasa=("precipitacao_mm_nasa", "sum"), inmet=("precipitacao_mm_inmet", "sum"))
            )
            referencia = (base_est["nasa"] / base_est["inmet"]).reindex(
                soma.index.get_level_values("codigo")
            )
            soma["relativo"] = (soma["nasa"] / soma["inmet"]).to_numpy() / referencia.to_numpy()
        else:
            validos = dados.dropna(subset=["temperatura_c", "tmed_c"]).assign(
                dif=lambda d: d["tmed_c"] - d["temperatura_c"]
            )
            soma = validos.groupby(["codigo", "semestre"]).agg(
                dif=("dif", "mean"), meses=("mes", "count")
            )
            referencia = validos[na_base.loc[validos.index]].groupby("codigo")["dif"].mean()
            soma["relativo"] = (
                soma["dif"].to_numpy()
                - referencia.reindex(soma.index.get_level_values("codigo")).to_numpy()
            )
        soma = soma[soma["meses"] >= min_meses_semestre]
        por_semestre = soma.groupby("semestre")["relativo"].agg(["median", "count"])
        linhas.append(por_semestre.rename(columns=lambda c, v=variavel: f"{v}_{c}"))
    return pd.concat(linhas, axis=1).reset_index()
