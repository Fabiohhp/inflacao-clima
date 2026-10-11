"""Análise exploratória de séries temporais: STL, raiz unitária, correlação cruzada
pré-branqueada e coerência wavelet."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews


def dessazonalizar_stl(serie: pd.Series, periodo: int = 12, robusto: bool = True) -> pd.DataFrame:
    """STL (Cleveland et al., 1990) de uma série mensal completa.

    Devolve observado, tendência, sazonal, resíduo e dessazonalizado (observado − sazonal).
    """
    if serie.isna().any():
        raise ValueError("a STL exige série sem lacunas")
    resultado = STL(serie, period=periodo, robust=robusto).fit()
    return pd.DataFrame(
        {
            "observado": serie,
            "tendencia": resultado.trend,
            "sazonal": resultado.seasonal,
            "residuo": resultado.resid,
            "dessazonalizado": serie - resultado.seasonal,
        }
    )


def testes_raiz_unitaria(serie: pd.Series, tendencia: str) -> dict[str, float]:
    """ADF (H0: raiz unitária, lags por AIC) e KPSS (H0: estacionária), com a mesma tendência.

    `tendencia` é "c" (constante) ou "ct" (constante e tendência). O p-valor do KPSS é
    tabelado entre 0,01 e 0,10; fora disso fica no limite (sem aviso).
    """
    valores = serie.dropna().to_numpy()
    adf = adfuller(valores, regression=tendencia, autolag="AIC", result_object=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        teste_kpss = kpss(valores, regression=tendencia, nlags="auto", result_object=False)
    return {
        "adf_estatistica": adf[0],
        "adf_p": adf[1],
        "adf_lags": adf[2],
        "kpss_estatistica": teste_kpss[0],
        "kpss_p": teste_kpss[1],
        "n": len(valores),
    }


def ordem_integracao(
    serie: pd.Series, tendencia_nivel: str, alfa: float = 0.05, max_ordem: int = 2
) -> tuple[str, pd.DataFrame]:
    """Ordem de integração pela regra conjunta ADF + KPSS.

    A série diferenciada d vezes é estacionária se o ADF rejeita a raiz unitária e o KPSS não
    rejeita a estacionariedade. O nível usa `tendencia_nivel`; as diferenças, constante. A
    ordem é o menor d estacionário; se nenhum até `max_ordem`, "> max_ordem". Quando os dois
    testes concordam na não estacionariedade do nível mas divergem numa diferença, a linha
    fica marcada como "ambíguo".
    """
    linhas = []
    atual = serie.dropna()
    for d in range(max_ordem + 1):
        regressao = tendencia_nivel if d == 0 else "c"
        resultado = testes_raiz_unitaria(atual, regressao)
        adf_rejeita = resultado["adf_p"] < alfa
        kpss_rejeita = resultado["kpss_p"] < alfa
        if adf_rejeita and not kpss_rejeita:
            conclusao = "estacionária"
        elif not adf_rejeita and kpss_rejeita:
            conclusao = "não estacionária"
        else:
            conclusao = "ambíguo"
        linhas.append({"diferenca": d, "tendencia": regressao, **resultado, "conclusao": conclusao})
        if conclusao == "estacionária":
            return f"I({d})", pd.DataFrame(linhas)
        atual = atual.diff().dropna()
    return f"> I({max_ordem})", pd.DataFrame(linhas)


def diagnostico_quebra_tendencia(serie: pd.Series) -> dict[str, object]:
    """Zivot-Andrews (H0: raiz unitária; H1: estacionária com uma quebra de nível) e ADF/KPSS
    com constante e tendência, para séries que a regra básica não classificou como I(0)."""
    valores = serie.dropna()
    za = zivot_andrews(valores.to_numpy(), regression="c", autolag="AIC")
    com_tendencia = testes_raiz_unitaria(valores, "ct")
    return {
        "za_p": za[1],
        "za_quebra": valores.index[za[4]],
        "adf_ct_p": com_tendencia["adf_p"],
        "kpss_ct_p": com_tendencia["kpss_p"],
    }


def leitura_integracao(
    ordem_base: str, tendencia_nivel: str, diagnostico: dict[str, object], alfa: float = 0.05
) -> str:
    """Leitura final da ordem de integração de uma série com constante (taxas, SPI, ENSO).

    Se a regra básica não deu I(0), mas o ADF com tendência rejeita e o KPSS com tendência não
    rejeita, a série é "I(0) em torno de tendência"; senão, se o Zivot-Andrews rejeita, é
    "I(0) com quebra (AAAA-MM)". Séries com tendência no nível ("ct") mantêm a ordem básica.
    """
    if ordem_base == "I(0)" or tendencia_nivel != "c":
        return ordem_base
    if diagnostico["adf_ct_p"] < alfa and diagnostico["kpss_ct_p"] >= alfa:
        return "I(0) em torno de tendência"
    if diagnostico["za_p"] < alfa:
        return f"I(0) com quebra ({diagnostico['za_quebra']:%Y-%m})"
    return ordem_base


def prebranquear(
    x: pd.Series, y: pd.Series, max_ordem: int = 12
) -> tuple[pd.Series, pd.Series, int]:
    """Pré-branqueamento de Box-Jenkins: AR(p) ajustado em x (p por AIC) e aplicado a x e y.

    Devolve (inovações de x, y filtrado, p). As duas séries são centradas antes do filtro.
    """
    dados = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    xc = dados["x"] - dados["x"].mean()
    yc = dados["y"] - dados["y"].mean()
    selecao = ar_select_order(xc.to_numpy(), maxlag=max_ordem, ic="aic", trend="n")
    ordens = selecao.ar_lags or []
    p = max(ordens) if ordens else 0
    if p == 0:
        return xc, yc, 0
    phi = np.zeros(p)
    ajuste = selecao.model.fit()
    for lag, coef in zip(ordens, ajuste.params, strict=True):
        phi[lag - 1] = coef

    def filtrar(serie: pd.Series) -> pd.Series:
        valores = serie.to_numpy()
        filtrado = valores.copy()
        for j in range(1, p + 1):
            filtrado[j:] -= phi[j - 1] * valores[:-j]
        return pd.Series(filtrado, index=serie.index).iloc[p:]

    return filtrar(xc), filtrar(yc), p


def ccf_prebranqueada(
    x: pd.Series, y: pd.Series, max_defasagem: int, alfa: float = 0.05
) -> pd.DataFrame:
    """Correlação entre x(t − k) e y(t), k = 0..max_defasagem, após o pré-branqueamento.

    Inclui a banda pontual (z_{1−α/2}/√n) e a de Bonferroni sobre as defasagens testadas.
    """
    ex, yf, p = prebranquear(x, y)
    n = len(ex)
    linhas = []
    for k in range(max_defasagem + 1):
        r = ex.shift(k).corr(yf)
        linhas.append({"defasagem": k, "r": r})
    tabela = pd.DataFrame(linhas)
    tabela["banda_pontual"] = stats.norm.ppf(1 - alfa / 2) / np.sqrt(n)
    tabela["banda_bonferroni"] = stats.norm.ppf(1 - alfa / (2 * (max_defasagem + 1))) / np.sqrt(n)
    tabela["ordem_ar"] = p
    tabela["n"] = n
    return tabela


def avaliar_defasagens(ccf: pd.DataFrame, janela: list[int] | None) -> dict[str, object]:
    """Compara a correlação cruzada com a janela de defasagens da hipótese.

    Sinal esperado: negativo (seca, SPI < 0, seguida de alta de preço). Classificação:
    "compatível" se a correlação mais negativa é significativa (banda pontual) e está na
    janela; "parcial" se há defasagem negativa significativa na janela, mas a mais forte fica
    fora; "fora da hipótese" se só há negativas significativas fora da janela; "sem sinal" se
    não há nenhuma; "exploratório" quando não há hipótese.
    """
    negativas = ccf[ccf["r"] < -ccf["banda_pontual"]]
    mais_negativa = ccf.loc[ccf["r"].idxmin()]
    significativa = mais_negativa["r"] < -mais_negativa["banda_pontual"]
    resumo = {
        "defasagens_negativas_signif": ", ".join(str(int(k)) for k in negativas["defasagem"]),
        "defasagem_mais_negativa": int(mais_negativa["defasagem"]),
        "r_mais_negativo": round(float(mais_negativa["r"]), 3),
        "signif_pontual": bool(significativa),
        "signif_bonferroni": bool(mais_negativa["r"] < -mais_negativa["banda_bonferroni"]),
    }
    if janela is None:
        resumo["avaliacao"] = "exploratório"
        return resumo
    na_janela = negativas["defasagem"].between(*janela)
    if significativa and janela[0] <= mais_negativa["defasagem"] <= janela[1]:
        resumo["avaliacao"] = "compatível"
    elif na_janela.any():
        resumo["avaliacao"] = "parcial"
    elif len(negativas):
        resumo["avaliacao"] = "fora da hipótese"
    else:
        resumo["avaliacao"] = "sem sinal"
    return resumo


def coerencia_wavelet(
    x: pd.Series, y: pd.Series, *, mc: int, semente: int, nivel: float = 0.95
) -> pd.DataFrame:
    """Coerência wavelet (Morlet) entre x e y, com significância por Monte Carlo (AR1).

    Devolve formato longo: mes, periodo (meses), coerencia, fase (rad), significativa (bool) e
    dentro_coi (fora da região afetada pelas bordas).
    """
    import pycwt

    dados = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    # O pycwt sorteia o ruído vermelho com o gerador global legado do numpy.
    np.random.seed(semente)  # noqa: NPY002
    wct, fase, coi, freq, sig = pycwt.wct(
        dados["x"].to_numpy(), dados["y"].to_numpy(), dt=1, dj=1 / 12, s0=2, J=-1, sig=True,
        significance_level=nivel, wavelet="morlet", normalize=True, mc_count=mc,
        progress=False, cache=False,
    )  # fmt: skip
    periodos = 1 / freq
    limiar = np.asarray(sig).reshape(-1, 1) if np.ndim(sig) else np.full((len(periodos), 1), np.inf)
    meses = np.tile(dados.index.to_numpy(), len(periodos))
    return pd.DataFrame(
        {
            "mes": meses,
            "periodo": np.repeat(periodos, len(dados)),
            "coerencia": wct.ravel(),
            "fase": fase.ravel(),
            "significativa": (wct >= limiar).ravel(),
            "dentro_coi": (periodos.reshape(-1, 1) <= coi.reshape(1, -1)).ravel(),
        }
    )
