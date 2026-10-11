"""Extremos diários: sequências de dias quentes e janelas críticas das culturas."""

from __future__ import annotations

from typing import Any

import pandas as pd

from clima_inflacao.espacial import cinturao_verde, cwi


def maior_sequencia(condicao: pd.Series, grupos: list[pd.Series]) -> pd.Series:
    """Maior número de dias consecutivos com `condicao` verdadeira em cada grupo.

    Os dados devem estar ordenados por grupo e por data, com os dias de um grupo contíguos.
    NaN conta como falso (interrompe a sequência).
    """
    verdadeiro = condicao.fillna(0).astype(bool)
    grupo_id = pd.Series(pd.factorize(pd.MultiIndex.from_arrays(grupos))[0], index=condicao.index)
    quebra = ~verdadeiro | grupo_id.ne(grupo_id.shift())
    contagem = verdadeiro.astype(int).groupby(quebra.cumsum()).cumsum()
    return contagem.groupby(grupos).max()


def safra_da_janela(datas: pd.Series, meses: list[int]) -> pd.Series:
    """Ano-safra de cada data numa janela de meses em ordem cronológica.

    Uma janela que cruza o ano (ex.: [12, 1, 2]) leva o ano em que termina: dezembro de 2020
    pertence à safra 2021.
    """
    cruza_o_ano = meses[0] > meses[-1]
    ano = datas.dt.year
    if cruza_o_ano:
        ano = ano + (datas.dt.month >= meses[0]).astype(int)
    return ano


def dias_esperados(safras: pd.Index, meses: list[int]) -> pd.Series:
    """Número de dias do calendário da janela em cada safra (considera anos bissextos)."""
    cruza_o_ano = meses[0] > meses[-1]
    totais = {}
    for safra in safras:
        totais[safra] = sum(
            pd.Period(
                f"{safra - 1 if cruza_o_ano and m >= meses[0] else safra}-{m:02d}"
            ).days_in_month
            for m in meses
        )
    return pd.Series(totais)


def janelas_criticas(diario: pd.DataFrame, culturas: dict[str, Any]) -> pd.DataFrame:
    """Extremos nas janelas críticas de cada cultura, por célula e safra.

    Nos meses críticos: dias de calor, maior sequência de calor, graus-dia de calor e chuva.
    Nos meses de geada (se houver): dias de geada. Safras com dias faltando ficam NaN.
    Espera as colunas celula_id, data, dia_calor, dia_geada, graus_dia_calor e precipitacao_mm.
    """
    resultados = []
    for cultura, cfg in culturas.items():
        partes = []
        for tipo, meses in [("criticos", cfg["meses_criticos"]), ("geada", cfg["meses_geada"])]:
            if not meses:
                continue
            janela = diario[diario["data"].dt.month.isin(meses)].sort_values(["celula_id", "data"])
            safra = safra_da_janela(janela["data"], meses).rename("safra")
            grupos = [janela["celula_id"], safra]
            if tipo == "criticos":
                resumo = janela.groupby(grupos).agg(
                    dias_calor=("dia_calor", "sum"),
                    graus_dia_calor=("graus_dia_calor", "sum"),
                    chuva_mm=("precipitacao_mm", "sum"),
                    n_dias=("precipitacao_mm", "count"),
                )
                resumo["max_seq_calor"] = maior_sequencia(janela["dia_calor"], grupos)
            else:
                resumo = janela.groupby(grupos).agg(
                    dias_geada=("dia_geada", "sum"), n_dias=("dia_geada", "count")
                )
            esperado = dias_esperados(resumo.index.get_level_values("safra").unique(), meses)
            completo = (
                resumo["n_dias"].to_numpy()
                == esperado.reindex(resumo.index.get_level_values("safra")).to_numpy()
            )
            resumo = resumo.drop(columns="n_dias").astype(float)
            resumo.loc[~completo] = float("nan")
            partes.append(resumo)
        tabela = pd.concat(partes, axis=1).reset_index().assign(cultura=cultura)
        resultados.append(tabela)
    colunas = ["celula_id", "cultura", "safra", "dias_calor", "max_seq_calor", "graus_dia_calor",
               "chuva_mm", "dias_geada"]  # fmt: skip
    return pd.concat(resultados, ignore_index=True).reindex(columns=colunas)


def anomalia_padronizada(
    mensal: pd.DataFrame, coluna: str, referencia: tuple[str, str]
) -> pd.Series:
    """(x − média) / desvio da mesma célula e mês do ano no período de referência.

    `mensal` tem celula_id, mes e `coluna`. Células e meses do ano sem desvio na referência
    (ou com desvio zero) ficam NaN.
    """
    mes_do_ano = mensal["mes"].dt.month
    na_ref = (mensal["mes"] >= pd.Timestamp(referencia[0])) & (
        mensal["mes"] <= pd.Timestamp(referencia[1])
    )
    chaves = [mensal["celula_id"], mes_do_ano]
    ref = mensal[coluna].where(na_ref)
    media = ref.groupby(chaves).transform("mean")
    desvio = ref.groupby(chaves).transform("std")
    return (mensal[coluna] - media) / desvio.where(desvio > 0)


def indicadores_limiar(
    mensal: pd.DataFrame,
    spi: str,
    limiar_seca: float,
    limiar_excesso: float,
    limiar_calor_z: float,
    referencia: tuple[str, str],
) -> pd.DataFrame:
    """Indicadores 0/1 por célula e mês: seca (SPI < limiar_seca), excesso (SPI >
    limiar_excesso) e calor (anomalia padronizada da Tmax média > limiar_calor_z).

    `mensal` tem celula_id, mes, `spi` e tmax_media_c. Onde o dado falta, o indicador é NaN,
    para que a agregação (média ou CWI) renormalize sobre as células com dado.
    """
    z_tmax = anomalia_padronizada(mensal, "tmax_media_c", referencia)
    valor_spi = mensal[spi]
    return pd.DataFrame(
        {
            "celula_id": mensal["celula_id"],
            "mes": mensal["mes"],
            "seca": (valor_spi < limiar_seca).astype(float).where(valor_spi.notna()),
            "excesso": (valor_spi > limiar_excesso).astype(float).where(valor_spi.notna()),
            "calor": (z_tmax > limiar_calor_z).astype(float).where(z_tmax.notna()),
        }
    )


def fracoes_em_extremo(
    indicadores: pd.DataFrame,
    micro_celula: pd.DataFrame,
    pesos: pd.DataFrame,
    culturas: list[str],
    selecao_cinturao: pd.DataFrame | None,
    tipos: list[str],
) -> pd.DataFrame:
    """Fração da produção (CWI de cada cultura) e das células do cinturão verde (média das
    áreas) em cada condição extrema, a partir dos indicadores 0/1 por célula.

    `micro_celula` liga cada microrregião (codigo) à sua célula; `pesos` são os de produção
    (defasados, com ano, ou fixos); `selecao_cinturao`, as células no raio de cada área (None
    para não calcular o cinturão). Devolve motor (cultura ou "cinturao"), mes e `tipos`.
    """
    partes = []
    if culturas:
        por_micro = micro_celula.merge(indicadores, on="celula_id").drop(columns="celula_id")
        agregado = cwi.cwi(por_micro, pesos[pesos["cultura"].isin(culturas)], tipos)
        partes.append(agregado.rename(columns={"cultura": "motor"})[["motor", "mes", *tipos]])
    if selecao_cinturao is not None:
        por_area = cinturao_verde.agregar(indicadores, selecao_cinturao, tipos)
        media = por_area.groupby("mes")[tipos].mean().reset_index()
        partes.append(media.assign(motor="cinturao")[["motor", "mes", *tipos]])
    return pd.concat(partes, ignore_index=True)
