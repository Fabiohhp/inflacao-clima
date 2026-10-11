"""Índice climático ponderado pela safra (CWI) e sua validação contra o rendimento."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def cwi(indicador: pd.DataFrame, pesos: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    """CWI(t) = soma_i w(i) x indicador(i,t), renormalizado sobre as microrregiões com dado.

    `indicador` tem codigo, mes e `colunas`; `pesos` tem cultura, codigo, peso e, se defasados,
    ano (aplicado aos meses daquele ano). Devolve cultura, mes, uma coluna por indicador e a
    cobertura (soma dos pesos com dado) de cada um.
    """
    dados = indicador.assign(ano=indicador["mes"].dt.year)
    chaves = ["codigo", "ano"] if "ano" in pesos.columns else ["codigo"]
    dados = dados.merge(pesos, on=chaves, how="inner")
    resultado = {}
    for coluna in colunas:
        com_dado = dados[coluna].notna()
        ponderado = (dados["peso"] * dados[coluna]).where(com_dado)
        peso_valido = dados["peso"].where(com_dado)
        grupos = [dados["cultura"], dados["mes"]]
        soma = ponderado.groupby(grupos).sum(min_count=1)
        cobertura = peso_valido.groupby(grupos).sum(min_count=1)
        resultado[coluna] = soma / cobertura
        resultado[f"cobertura_{coluna}"] = cobertura
    return pd.DataFrame(resultado).reset_index()


def anomalia_rendimento(rendimento: pd.Series, anos: tuple[int, int]) -> pd.Series:
    """Desvio (%) do rendimento em relação à tendência linear do log, ajustada em `anos`."""
    serie = rendimento.dropna()
    serie = serie[(serie.index >= anos[0]) & (serie.index <= anos[1])]
    coef = np.polyfit(serie.index, np.log(serie.to_numpy()), 1)
    tendencia = np.polyval(coef, serie.index)
    return pd.Series(100 * (np.exp(np.log(serie.to_numpy()) - tendencia) - 1), index=serie.index)


def validar(
    cwi_brasil: pd.DataFrame,
    rendimento: pd.DataFrame,
    criterios: list[dict[str, Any]],
    anos: tuple[int, int],
    coluna: str = "spi_3",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica o critério de aceite do CWI a cada cultura.

    `cwi_brasil` tem cultura, mes e `coluna`; `rendimento` tem ano, produto e rendimento_kg_ha
    (Brasil). Devolve (resumo por cultura, tabela safra a safra com CWI e anomalia).
    """
    resumos, tabelas = [], []
    for criterio in criterios:
        cultura = criterio["cultura"]
        serie = cwi_brasil[
            (cwi_brasil["cultura"] == cultura)
            & (cwi_brasil["mes"].dt.month == criterio["mes_final"])
        ]
        indice = serie.set_index(serie["mes"].dt.year)[coluna]
        rend = rendimento[rendimento["produto"] == cultura].set_index("ano")["rendimento_kg_ha"]
        anomalia = anomalia_rendimento(rend, anos)
        tabela = pd.DataFrame({"cwi": indice, "anomalia_rendimento_pct": anomalia}).dropna()
        tabela = tabela.rename_axis("safra").reset_index().assign(cultura=cultura)
        tabela["ano_de_quebra"] = tabela["safra"].isin(criterio["anos_quebra"])
        r, p = stats.pearsonr(tabela["cwi"], tabela["anomalia_rendimento_pct"])
        quebras = tabela[tabela["ano_de_quebra"]]
        resumos.append(
            {
                "cultura": cultura,
                "safras": f"{tabela['safra'].min()}-{tabela['safra'].max()}",
                "n": len(tabela),
                "correlacao": round(r, 3),
                "p_valor": round(p, 4),
                "cwi_nos_anos_de_quebra": "; ".join(
                    f"{s}: {v:.2f}" for s, v in zip(quebras["safra"], quebras["cwi"], strict=True)
                ),
                "passou": bool((quebras["cwi"] < 0).all() and r > 0 and p < 0.05),
            }
        )
        tabelas.append(tabela)
    return pd.DataFrame(resumos), pd.concat(tabelas, ignore_index=True)
