"""Representatividade de um preço regional: o Paraná chega perto do resto do Brasil?

As comparações usam variações em log (× 100) e descontam a inflação geral (IPCA): as oscilações
da inflação, comuns a todos os preços, inflam a correlação nominal sem dizer nada sobre o
produto.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


@dataclass(frozen=True)
class Criterios:
    """Cortes registrados no settings (`precos_atacado.representatividade.criterios`)."""

    correlacao_real_min: float
    diferenca_media_max_fracao_inflacao: float
    divergencia_acumulada_max_fracao_inflacao: float


def regularizar(niveis: pd.DataFrame) -> pd.DataFrame:
    """Reindexa para uma grade sem buracos (meses ou anos), para que `diff(k)` seja k períodos.

    Índice de datas → todos os meses entre o primeiro e o último; índice inteiro → todos os anos.
    """
    indice = niveis.index
    if isinstance(indice, pd.DatetimeIndex):
        if (indice.day != 1).any():
            raise ValueError("datas mensais devem estar no 1º dia do mês")
        grade = pd.date_range(indice.min(), indice.max(), freq="MS", name=indice.name)
    elif pd.api.types.is_integer_dtype(indice):
        grade = pd.RangeIndex(indice.min(), indice.max() + 1, name=indice.name)
    else:
        raise ValueError("índice deve ser de datas mensais ou de anos inteiros")
    return niveis.reindex(grade)


def maior_trecho_continuo(serie: pd.Series) -> pd.Series:
    """Maior sequência sem faltantes de uma série já regularizada."""
    valida = serie.notna().to_numpy()
    melhor, inicio_melhor, inicio = 0, 0, None
    for i, ok in enumerate(np.append(valida, False)):
        if ok and inicio is None:
            inicio = i
        elif not ok and inicio is not None:
            if i - inicio > melhor:
                melhor, inicio_melhor = i - inicio, inicio
            inicio = None
    return serie.iloc[inicio_melhor : inicio_melhor + melhor]


def _tempo_em_anos(indice: pd.Index) -> np.ndarray:
    if isinstance(indice, pd.DatetimeIndex):
        return (indice.year + (indice.month - 1) / 12).to_numpy(dtype=float)
    return indice.to_numpy(dtype=float)


def comparar(
    pr: pd.Series,
    referencia: pd.Series,
    geral: pd.Series,
    *,
    defasagem: int = 1,
    hac_lags: int | None = None,
    adf_min_obs: int = 24,
) -> dict[str, float]:
    """Métricas de proximidade entre o preço do PR e o de referência (níveis no mesmo índice).

    - variações em `defasagem` períodos, 100 × Δlog; reais = nominal − inflação geral (`geral`
      é o índice geral de preços);
    - correlação real e nominal; inclinação da variação real da referência sobre a do PR (1 =
      o PR não atenua nem amplia), com erro HAC (`hac_lags`, padrão = `defasagem`);
    - diferença média = média de |variação PR − variação referência|, a comparar com a inflação
      geral média nas mesmas variações;
    - divergência acumulada = log(PR/ref) no último período comum − no primeiro (× 100), a
      comparar com a inflação geral acumulada entre os mesmos períodos;
    - tendência do log da razão (pontos de log por ano, erro HAC) e p-valor do ADF (com
      constante) no maior trecho contínuo da razão, se tiver ao menos `adf_min_obs` períodos.
    """
    niveis = pd.DataFrame({"pr": pr, "ref": referencia, "geral": geral}).astype(float)
    if (niveis <= 0).any().any():
        raise ValueError("preços e índice geral devem ser positivos")
    logs = 100 * np.log(regularizar(niveis))
    variacoes = logs.diff(defasagem).dropna()
    if len(variacoes) < 5:
        raise ValueError(f"só {len(variacoes)} variações em comum")
    real_pr = variacoes["pr"] - variacoes["geral"]
    real_ref = variacoes["ref"] - variacoes["geral"]
    lags = defasagem if hac_lags is None else hac_lags
    inclinacao = sm.OLS(real_ref, sm.add_constant(real_pr)).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )

    razao = (logs["pr"] - logs["ref"]).dropna()
    primeiro, ultimo = razao.index[0], razao.index[-1]
    tempo = _tempo_em_anos(razao.index)
    tendencia = sm.OLS(razao.to_numpy(), sm.add_constant(tempo)).fit(
        cov_type="HAC", cov_kwds={"maxlags": max(lags, 1)}
    )
    trecho = maior_trecho_continuo(logs["pr"] - logs["ref"])
    adf_p = np.nan  # razão constante (mesma série) ou trecho curto: sem teste
    if len(trecho) >= adf_min_obs and np.ptp(trecho.to_numpy()) > 1e-9:
        adf_p = adfuller(trecho.to_numpy(), regression="c", autolag="AIC", result_object=False)[1]

    return {
        "n_variacoes": len(variacoes),
        "inicio": primeiro,
        "fim": ultimo,
        "correlacao_real": float(np.corrcoef(real_pr, real_ref)[0, 1]),
        "correlacao_nominal": float(np.corrcoef(variacoes["pr"], variacoes["ref"])[0, 1]),
        "inclinacao": float(inclinacao.params.iloc[1]),
        "inclinacao_ep": float(inclinacao.bse.iloc[1]),
        "diferenca_media": float((variacoes["pr"] - variacoes["ref"]).abs().mean()),
        "inflacao_media": float(variacoes["geral"].mean()),
        "divergencia_acumulada": float(razao.iloc[-1] - razao.iloc[0]),
        "inflacao_acumulada": float(logs.loc[ultimo, "geral"] - logs.loc[primeiro, "geral"]),
        "tendencia_razao_ano": float(tendencia.params[1]),
        "tendencia_p": float(tendencia.pvalues[1]),
        "adf_p": float(adf_p),
        "n_adf": len(trecho),
    }


def avaliar(metricas: dict[str, float], criterios: Criterios) -> dict[str, bool]:
    """Aplica os três critérios registrados; `chega_perto` exige os três."""
    resultado = {
        "ok_correlacao": metricas["correlacao_real"] >= criterios.correlacao_real_min,
        "ok_diferenca": metricas["diferenca_media"]
        <= criterios.diferenca_media_max_fracao_inflacao * metricas["inflacao_media"],
        "ok_divergencia": abs(metricas["divergencia_acumulada"])
        <= criterios.divergencia_acumulada_max_fracao_inflacao * metricas["inflacao_acumulada"],
    }
    resultado["chega_perto"] = all(resultado.values())
    return resultado


def retrato_uf(precos: pd.DataFrame, uf: str = "PR") -> dict[str, float]:
    """Retrato descritivo de um produto entre UFs (`precos`: meses × UFs, já de um só nível).

    Usa só as UFs com todos os meses (composição fixa). Devolve o número de UFs, a razão de
    nível da UF sobre a mediana das outras (média dos meses, em %), o percentil da UF entre
    todas e a correlação das variações mensais da UF com as da mediana das outras.
    """
    completas = precos.dropna(axis=1, how="any")
    if uf not in completas.columns:
        raise ValueError(f"{uf} não tem todos os meses")
    logs = np.log(completas)
    outras = logs.drop(columns=uf).median(axis=1)
    variacoes = pd.DataFrame({"uf": logs[uf], "outras": outras}).diff().dropna()
    percentil = completas.rank(axis=1, pct=True)[uf].mean()
    return {
        "n_ufs": completas.shape[1],
        "n_meses": len(completas),
        "razao_nivel_pct": float(100 * (np.exp((logs[uf] - outras).mean()) - 1)),
        "percentil_medio": float(percentil),
        "correlacao_variacoes": float(variacoes.corr().iloc[0, 1]),
    }


def concordancia(a: pd.Series, b: pd.Series) -> dict[str, float]:
    """Concordância de duas fontes para o mesmo preço nos meses comuns: correlação das variações
    mensais (em log) e coeficiente de variação da razão a/b (0 = mesma série em outra unidade)."""
    comum = pd.DataFrame({"a": a, "b": b}).dropna()
    razao = comum["a"] / comum["b"]
    variacoes = np.log(regularizar(comum)).diff().dropna()
    return {
        "n_meses": len(comum),
        "correlacao_variacoes": float(variacoes.corr().iloc[0, 1]),
        "razao_cv_pct": float(100 * razao.std() / razao.mean()),
    }


def indice_pesos_fixos(precos: pd.DataFrame, pesos: pd.Series) -> pd.Series:
    """Índice de preço de um conjunto de UFs com pesos fixos (`precos`: anos × UFs).

    Só entram as UFs com preço positivo em todos os anos; os pesos são renormalizados entre
    elas. A variação do índice é a média ponderada das variações em log de cada UF, e o
    índice começa em 1. Devolve também, em `attrs`, as UFs usadas e o peso que ficou de fora.
    """
    completas = precos.loc[:, (precos > 0).all() & precos.notna().all()]
    usadas = [uf for uf in completas.columns if pesos.get(uf, 0) > 0]
    w = pesos[usadas] / pesos[usadas].sum()
    variacao = np.log(completas[usadas]).diff().fillna(0) @ w
    indice = np.exp(variacao.cumsum())
    indice.attrs = {"ufs": usadas, "peso_fora": float(1 - pesos[usadas].sum() / pesos.sum())}
    return indice
