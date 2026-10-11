"""Previsão pseudo fora da amostra com regressões diretas e janela expansiva (ou móvel).

Na origem t e no horizonte h, o alvo é a inflação acumulada dos h meses seguintes,
100 x [log p(t+h) − log p(t)]. Os regressores usam só a informação do fim de t: constante,
dummies do mês de t, inflação mensal de t a t − (lags_inflacao − 1) e cada preditor de t a
t − (lags_preditor − 1). Em cada origem, o modelo é reestimado só com as origens s cujo alvo já
foi observado (s + h <= t). Testes de igualdade de acurácia: Clark e West (2007), para modelos
aninhados, e Diebold e Mariano (1995) com a correção de Harvey, Leybourne e Newbold (1997), para
não aninhados. "Pseudo": os dados são os da versão final, não os disponíveis em tempo real.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class EspecificacaoPrevisao:
    """Regressão direta de previsão. `preditores` são colunas já defasadas pela disponibilidade
    (o valor na linha t é o conhecido no fim de t)."""

    log_preco: str
    preditores: list[str] = field(default_factory=list)
    lags_inflacao: int = 12
    lags_preditor: int = 3
    dummies_mes: bool = True


def regressores(dados: pd.DataFrame, espec: EspecificacaoPrevisao) -> pd.DataFrame:
    """Matriz de regressores na origem t (linha t usa só dados até t)."""
    inflacao = (100 * dados[espec.log_preco]).diff()
    colunas = {"const": pd.Series(1.0, index=dados.index)}
    for lag in range(espec.lags_inflacao):
        colunas[f"inflacao_l{lag}"] = inflacao.shift(lag)
    for preditor in espec.preditores:
        for lag in range(espec.lags_preditor):
            colunas[f"{preditor}_l{lag}"] = dados[preditor].shift(lag)
    x = pd.DataFrame(colunas, index=dados.index)
    if espec.dummies_mes:
        meses = pd.get_dummies(dados.index.month, prefix="mes", drop_first=True, dtype=float)
        x = pd.concat([x, meses.set_index(dados.index)], axis=1)
    return x


def alvo(dados: pd.DataFrame, log_preco: str, h: int) -> pd.Series:
    """100 x [log p(t+h) − log p(t)], indexado pela origem t."""
    log_p = 100 * dados[log_preco]
    return log_p.shift(-h) - log_p


def prever(
    dados: pd.DataFrame,
    espec: EspecificacaoPrevisao,
    h: int,
    inicio: pd.Timestamp,
    primeira_origem: pd.Timestamp,
    fim: pd.Timestamp,
    janela_movel: int | None = None,
) -> pd.DataFrame:
    """Previsões diretas em cada origem t de `primeira_origem` a `fim` − h.

    `dados` é mensal, sem lacunas no índice, e inclui os meses anteriores a `inicio` usados nas
    defasagens. Na origem t, a estimação usa as origens s de `inicio` (ou, com `janela_movel`,
    das últimas `janela_movel` origens) a t − h. Saída: origem, horizonte, alvo, previsao, erro
    (alvo − previsão) e n_estimacao. Levanta ValueError se houver faltante.
    """
    x, y = regressores(dados, espec), alvo(dados, espec.log_preco, h)
    meses = dados.index
    matriz, valores = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
    ultima_origem = fim - pd.DateOffset(months=h)
    origens = meses[(meses >= primeira_origem) & (meses <= ultima_origem)]
    linhas = []
    for origem in origens:
        ultimo = origem - pd.DateOffset(months=h)
        estimacao = np.flatnonzero((meses >= inicio) & (meses <= ultimo))
        if janela_movel is not None:
            estimacao = estimacao[-janela_movel:]
        xe, ye = matriz[estimacao], valores[estimacao]
        posicao = meses.get_loc(origem)
        x0 = matriz[posicao]
        if np.isnan(xe).any() or np.isnan(ye).any() or np.isnan(x0).any():
            raise ValueError(f"origem {origem:%Y-%m}, h={h}: faltantes na estimação ou na origem")
        beta = np.linalg.lstsq(xe, ye, rcond=None)[0]
        previsao = float(x0 @ beta)
        linhas.append(
            {"origem": origem, "horizonte": h, "alvo": valores[posicao], "previsao": previsao,
             "erro": valores[posicao] - previsao, "n_estimacao": len(ye)}
        )  # fmt: skip
    resultado = pd.DataFrame(linhas)
    if resultado["alvo"].isna().any():
        raise ValueError(f"h={h}: alvo faltante dentro da avaliação")
    return resultado


def variancia_longo_prazo(x: np.ndarray, lags: int) -> float:
    """Variância de longo prazo de Newey-West (pesos de Bartlett) de uma série."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    variancia = x @ x / n
    for k in range(1, lags + 1):
        variancia += 2 * (1 - k / (lags + 1)) * (x[k:] @ x[:-k]) / n
    return float(variancia)


def clark_west(alvo: np.ndarray, prev_restrito: np.ndarray, prev_amplo: np.ndarray,
               h: int) -> dict:  # fmt: skip
    """Teste de Clark e West (2007): o modelo amplo (que aninha o restrito) prevê melhor?

    Estatística t da média de f = e_r² − [e_a² − (ŷ_r − ŷ_a)²], com variância de longo prazo de
    Newey-West (h − 1 defasagens, pela sobreposição dos alvos); p-valor unilateral (normal).
    """
    alvo, prev_r, prev_a = (np.asarray(v, dtype=float) for v in (alvo, prev_restrito, prev_amplo))
    erro_r, erro_a = alvo - prev_r, alvo - prev_a
    f = erro_r**2 - (erro_a**2 - (prev_r - prev_a) ** 2)
    n = len(f)
    estatistica = f.mean() / np.sqrt(variancia_longo_prazo(f, h - 1) / n)
    return {"estatistica": float(estatistica), "p_valor": float(stats.norm.sf(estatistica)),
            "eqm_restrito": float(np.mean(erro_r**2)), "eqm_amplo": float(np.mean(erro_a**2)),
            "n": n}  # fmt: skip


def diebold_mariano(erro_a: np.ndarray, erro_b: np.ndarray, h: int) -> dict:
    """Diebold-Mariano com a correção de Harvey, Leybourne e Newbold (1997), bilateral.

    d = e_a² − e_b²; estatística positiva = o modelo a erra mais (b é melhor). Variância de
    longo prazo de Newey-West (h − 1 defasagens); distribuição t com n − 1 graus de liberdade.
    """
    erro_a, erro_b = np.asarray(erro_a, dtype=float), np.asarray(erro_b, dtype=float)
    d = erro_a**2 - erro_b**2
    n = len(d)
    dm = d.mean() / np.sqrt(variancia_longo_prazo(d, h - 1) / n)
    estatistica = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return {"estatistica": float(estatistica),
            "p_valor": float(2 * stats.t.sf(abs(estatistica), df=n - 1)),
            "eqm_a": float(np.mean(erro_a**2)), "eqm_b": float(np.mean(erro_b**2)),
            "n": n}  # fmt: skip


def r2_fora_amostra(erro_modelo: np.ndarray, erro_benchmark: np.ndarray) -> float:
    """1 − EQM do modelo / EQM do benchmark (positivo = o modelo erra menos)."""
    erro_modelo, erro_benchmark = np.asarray(erro_modelo), np.asarray(erro_benchmark)
    return float(1 - np.sum(erro_modelo**2) / np.sum(erro_benchmark**2))


def r2_defasagens(y: pd.Series, x: pd.Series, lags: int) -> dict:
    """R² e R² ajustado da regressão de `y` numa constante e em `x` de t a t − lags (só os
    meses com tudo observado), e a correlação contemporânea. Descritivo (sem inferência)."""
    colunas = pd.concat({f"l{k}": x.shift(k) for k in range(lags + 1)}, axis=1)
    dados = pd.concat([y.rename("y"), colunas], axis=1).dropna()
    matriz = np.column_stack([np.ones(len(dados)), dados.drop(columns="y").to_numpy()])
    beta = np.linalg.lstsq(matriz, dados["y"].to_numpy(), rcond=None)[0]
    residuo = dados["y"].to_numpy() - matriz @ beta
    total = dados["y"] - dados["y"].mean()
    n, k = matriz.shape
    r2 = 1 - residuo @ residuo / (total @ total)
    return {"r2": float(r2), "r2_ajustado": float(1 - (1 - r2) * (n - 1) / (n - k)),
            "corr_t": float(dados["y"].corr(dados["l0"])), "n": n}  # fmt: skip


def alvo_cruza_periodo(origens: pd.Series, horizontes: pd.Series, inicio: pd.Timestamp,
                       fim: pd.Timestamp) -> pd.Series:  # fmt: skip
    """True se a janela do alvo (meses t + 1 a t + h) tem algum mês em [inicio, fim]."""
    meses = pd.to_datetime(pd.Series(origens)).dt.to_period("M").reset_index(drop=True)
    primeiro = meses + 1
    ultimo = meses + np.asarray(horizontes, dtype=int)
    cruza = (primeiro <= pd.Period(fim, "M")) & (ultimo >= pd.Period(inicio, "M"))
    return pd.Series(cruza.to_numpy(), index=pd.Series(origens).index)


def avaliar(previsoes: pd.DataFrame, comparacoes: dict[str, str], indices: list[str],
            nivel: float) -> tuple[pd.DataFrame, pd.DataFrame]:  # fmt: skip
    """Avaliação por grupo × horizonte, como na Parte 6c.

    `previsoes` tem dependente, horizonte, modelo, origem, alvo, previsao e erro, com as mesmas
    origens em todos os modelos de um grupo × horizonte. `comparacoes` mapeia modelo amplo →
    restrito (Clark-West); cada índice ENSO de `indices` é comparado ao "local" por
    Diebold-Mariano. Devolve (avaliação com R² fora da amostra e Clark-West, Diebold-Mariano).
    """
    linhas, linhas_dm = [], []
    for (dep, h), grupo in previsoes.groupby(["dependente", "horizonte"]):
        por_modelo = {m: g.set_index("origem").sort_index() for m, g in grupo.groupby("modelo")}
        bench = por_modelo["benchmark"]
        if not all(g.index.equals(bench.index) for g in por_modelo.values()):
            raise ValueError(f"{dep}, h={h}: modelos com origens diferentes")
        linhas.append({"dependente": dep, "horizonte": h, "modelo": "benchmark",
                       "eqm": np.mean(bench["erro"] ** 2), "r2_oos_pct": 0.0,
                       "n": len(bench)})  # fmt: skip
        for modelo, restrito in comparacoes.items():
            amplo, menor = por_modelo[modelo], por_modelo[restrito]
            cw = clark_west(amplo["alvo"], menor["previsao"], amplo["previsao"], h)
            linhas.append(
                {"dependente": dep, "horizonte": h, "modelo": modelo, "eqm": cw["eqm_amplo"],
                 "r2_oos_pct": 100 * r2_fora_amostra(amplo["erro"], bench["erro"]),
                 "comparado_com": restrito,
                 "r2_oos_comparado_pct": 100 * r2_fora_amostra(amplo["erro"], menor["erro"]),
                 "cw_estatistica": cw["estatistica"], "cw_p": cw["p_valor"], "n": cw["n"]}
            )  # fmt: skip
        for indice in indices:
            dm = diebold_mariano(por_modelo[indice]["erro"], por_modelo["local"]["erro"], h)
            if dm["p_valor"] < nivel:
                classe = "ENSO superior" if dm["estatistica"] < 0 else "local superior"
            else:
                classe = "sem diferença"
            linhas_dm.append(
                {"dependente": dep, "horizonte": h, "indice": indice, "eqm_enso": dm["eqm_a"],
                 "eqm_local": dm["eqm_b"], "razao_eqm": dm["eqm_a"] / dm["eqm_b"],
                 "dm_estatistica": dm["estatistica"], "dm_p": dm["p_valor"], "classe": classe,
                 "n": dm["n"]}
            )  # fmt: skip
    avaliacao = pd.DataFrame(linhas)
    avaliacao["vence_10pct"] = avaliacao["cw_p"] < nivel
    return avaliacao, pd.DataFrame(linhas_dm)
