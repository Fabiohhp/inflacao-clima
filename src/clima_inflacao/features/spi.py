"""Índice Padronizado de Precipitação (SPI; McKee et al., 1993).

A chuva acumulada em k meses é ajustada, para cada mês do ano, a uma distribuição gama mista:
probabilidade q de zero e gama nos valores positivos, com parâmetros da aproximação de Thom
(1958) estimados só no período de referência. O SPI é o quantil normal padrão da
probabilidade acumulada. Valores iguais a zero recebem H = q/2, o centro da massa de zeros
(Stagge et al., 2015).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def parametros_thom(positivos: np.ndarray) -> tuple[float, float]:
    """Forma (alfa) e escala (beta) da gama pela aproximação de Thom à máxima verossimilhança."""
    media = positivos.mean()
    a = np.log(media) - np.log(positivos).mean()
    alfa = (1 + np.sqrt(1 + 4 * a / 3)) / (4 * a)
    return alfa, media / alfa


def spi_serie(
    chuva_mensal: pd.Series,
    escala: int,
    referencia: tuple[pd.Timestamp, pd.Timestamp],
    *,
    min_positivos: int = 10,
    limite_abs: float = 3.09,
) -> pd.Series:
    """SPI de uma série mensal (índice = 1º dia de cada mês, sem meses faltando nas linhas).

    O acumulado de `escala` meses termina no mês do índice. Meses do ano com menos de
    `min_positivos` acumulados positivos na referência ficam NaN.
    """
    acumulado = chuva_mensal.rolling(escala, min_periods=escala).sum()
    meses = acumulado.index
    na_referencia = (meses >= referencia[0]) & (meses <= referencia[1])
    resultado = pd.Series(np.nan, index=meses, name=f"spi_{escala}")
    for mes_do_ano in range(1, 13):
        deste_mes = meses.month == mes_do_ano
        ref = acumulado[deste_mes & na_referencia].dropna().to_numpy()
        positivos = ref[ref > 0]
        if len(positivos) < min_positivos:
            continue
        q = (ref == 0).mean()
        alfa, beta = parametros_thom(positivos)
        x = acumulado[deste_mes].to_numpy()
        prob = np.where(x > 0, q + (1 - q) * stats.gamma.cdf(x, alfa, scale=beta), q / 2)
        valores = np.clip(stats.norm.ppf(prob), -limite_abs, limite_abs)
        resultado[deste_mes] = np.where(np.isnan(x), np.nan, valores)
    return resultado


def spi_painel(
    mensal: pd.DataFrame,
    escalas: list[int],
    referencia: tuple[pd.Timestamp, pd.Timestamp],
    *,
    min_positivos: int = 10,
    limite_abs: float = 3.09,
) -> pd.DataFrame:
    """SPI de cada célula (colunas celula_id, mes, precipitacao_mm), em grade mensal completa."""
    partes = []
    for celula_id, grupo in mensal.groupby("celula_id"):
        serie = grupo.set_index("mes")["precipitacao_mm"].sort_index()
        grade = pd.date_range(serie.index.min(), serie.index.max(), freq="MS", name="mes")
        serie = serie.reindex(grade)
        colunas = {
            f"spi_{k}": spi_serie(serie, k, referencia, min_positivos=min_positivos,
                                  limite_abs=limite_abs)
            for k in escalas
        }  # fmt: skip
        partes.append(pd.DataFrame(colunas).reset_index().assign(celula_id=celula_id))
    return pd.concat(partes, ignore_index=True)
