"""Pesos de produção por microrregião e cultura, a partir da PAM.

w(i,k) = Produção(i,k) / soma_j Produção(j,k). Os pesos principais são defasados (média dos
anos t-1 a t-3), para não usar a safra contemporânea, que é afetada pelo próprio choque.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def producao_microrregioes(pam: pd.DataFrame, culturas: list[str]) -> pd.DataFrame:
    """Produção (t) por ano, microrregião (código N9) e cultura, em formato largo (ano x código)."""
    micro = pam[(pam["nivel"] == "n9") & pam["produto"].isin(culturas)]
    return micro.rename(columns={"territorio_codigo": "codigo", "produto": "cultura"})[
        ["cultura", "ano", "codigo", "quantidade_t"]
    ]


def _normalizar(media: pd.Series) -> pd.Series:
    """Divide pela soma; microrregiões sem dado nenhum na janela recebem peso 0."""
    media = media.fillna(0.0)
    total = media.sum()
    if total <= 0:
        raise ValueError("produção total nula: pesos indefinidos")
    return media / total


def pesos_defasados(
    producao: pd.DataFrame, anos_aplicacao: list[int], defasagem: tuple[int, int] = (1, 3)
) -> pd.DataFrame:
    """Peso de cada microrregião no ano t = média da produção de t-1 a t-3 (anos disponíveis).

    Anos sem nenhum ano anterior com dado (início da série) usam a média dos 3 primeiros anos
    com dado; a coluna `origem` registra qual regra foi usada.
    """
    linhas = []
    for cultura, grupo in producao.groupby("cultura"):
        largo = grupo.pivot(index="ano", columns="codigo", values="quantidade_t")
        # Só contam os anos com produção válida (ex.: banana e laranja até 2000 ficam ausentes
        # na PAM processada, porque a unidade publicada não era tonelada).
        anos = largo.index[largo.sum(axis=1, min_count=1) > 0]
        for ano in anos_aplicacao:
            janela = [a for a in range(ano - defasagem[1], ano - defasagem[0] + 1) if a in anos]
            origem = "defasado"
            if not janela:
                janela = list(anos[:3])
                origem = "inicio_da_serie"
            peso = _normalizar(largo.loc[janela].mean(skipna=True))
            linhas.append(
                pd.DataFrame(
                    {
                        "cultura": cultura,
                        "ano": ano,
                        "codigo": peso.index,
                        "peso": peso.to_numpy(),
                        "origem": origem,
                    }
                )  # fmt: skip
            )
    return pd.concat(linhas, ignore_index=True)


def pesos_fixos(producao: pd.DataFrame, inicio: int, fim: int) -> pd.DataFrame:
    """Peso de cada microrregião = média da produção no período-base [inicio, fim]."""
    linhas = []
    for cultura, grupo in producao[producao["ano"].between(inicio, fim)].groupby("cultura"):
        largo = grupo.pivot(index="ano", columns="codigo", values="quantidade_t")
        peso = _normalizar(largo.mean(skipna=True))
        linhas.append(
            pd.DataFrame({"cultura": cultura, "codigo": peso.index, "peso": peso.to_numpy()})
        )
    return pd.concat(linhas, ignore_index=True)


def restringir(pesos: pd.DataFrame, codigos: list[str]) -> pd.DataFrame:
    """Pesos renormalizados dentro de um subconjunto de microrregiões (ex.: uma região)."""
    chaves = [c for c in ["cultura", "ano"] if c in pesos.columns]
    dentro = pesos[pesos["codigo"].isin(codigos)].copy()
    total = dentro.groupby(chaves)["peso"].transform("sum")
    dentro["peso"] = np.where(total > 0, dentro["peso"] / total, np.nan)
    return dentro.dropna(subset=["peso"])
