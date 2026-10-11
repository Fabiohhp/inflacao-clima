"""Local Projections (Jordà, 2005) com erros HAC de Newey-West.

Para cada horizonte h, regride 100 x [log p(t+h) − log p(t−1)] no choque em t, nas
defasagens do choque e da inflação mensal, em controles opcionais, em dummies de mês e numa
tendência linear. O coeficiente do choque em t é a resposta acumulada do preço em h. Na versão
marginal, a dependente é a inflação do mês t+h, 100 x [log p(t+h) − log p(t+h−1)]. A versão em
painel (`estimar_painel`) empilha unidades (áreas do IPCA) com erros de Driscoll-Kraay.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


@dataclass
class EspecificacaoLP:
    """Desenho de uma Local Projection.

    `log_preco` é a coluna do log do índice (nível); `choques`, as colunas dos choques (já com
    o sinal desejado); `controles`, colunas em log cujas variações entram só defasadas;
    `contrastes`, pares de choques (a, b) cuja diferença a − b é testada em cada horizonte.
    """

    log_preco: str
    choques: list[str]
    horizontes: range
    lags_dependente: int = 12
    lags_choque: int = 12
    controles: list[str] = field(default_factory=list)
    lags_controles: int = 6
    dummies_mes: bool = True
    tendencia: bool = True
    hac_lags_extra: int = 1
    niveis_confianca: tuple[float, ...] = (0.68, 0.90)
    acumulada: bool = True
    contrastes: list[tuple[str, str]] = field(default_factory=list)


def montar_regressores(dados: pd.DataFrame, espec: EspecificacaoLP) -> pd.DataFrame:
    """Matriz de regressores (sem a dependente), indexada pelo mês t."""
    log_p = 100 * dados[espec.log_preco]
    inflacao = log_p.diff()
    colunas = {}
    for choque in espec.choques:
        for lag in range(espec.lags_choque + 1):
            colunas[f"{choque}_l{lag}"] = dados[choque].shift(lag)
    for lag in range(1, espec.lags_dependente + 1):
        colunas[f"inflacao_l{lag}"] = inflacao.shift(lag)
    for controle in espec.controles:
        variacao = 100 * dados[controle].diff()
        for lag in range(1, espec.lags_controles + 1):
            colunas[f"d_{controle}_l{lag}"] = variacao.shift(lag)
    regressores = pd.DataFrame(colunas, index=dados.index)
    if espec.dummies_mes:
        meses = pd.get_dummies(dados.index.month, prefix="mes", drop_first=True, dtype=float)
        regressores = pd.concat([regressores, meses.set_index(dados.index)], axis=1)
    if espec.tendencia:
        regressores["tendencia"] = np.arange(len(dados), dtype=float) / 12
    return sm.add_constant(regressores)


def dependente(dados: pd.DataFrame, log_preco: str, h: int, acumulada: bool = True) -> pd.Series:
    """100 x [log p(t+h) − log p(t−1)] ou, se não acumulada, 100 x [log p(t+h) − log p(t+h−1)],
    indexada por t. Em h = 0 as duas coincidem."""
    log_p = 100 * dados[log_preco]
    base = log_p.shift(1) if acumulada else log_p.shift(-h + 1)
    return log_p.shift(-h) - base


def _linha(nome: str, h: int, coef: float, ep: float, p_valor: float, n: int,
           niveis: tuple[float, ...]) -> dict:  # fmt: skip
    linha = {"choque": nome, "horizonte": h, "coef": coef, "ep": ep, "p_valor": p_valor, "n": n}
    for nivel in niveis:
        z = stats.norm.ppf(0.5 + nivel / 2)
        rotulo = int(round(100 * nivel))
        linha[f"li_{rotulo}"] = coef - z * ep
        linha[f"ls_{rotulo}"] = coef + z * ep
    return linha


def estimar(
    dados: pd.DataFrame,
    espec: EspecificacaoLP,
    inicio: pd.Timestamp,
    fim: pd.Timestamp,
    excluir_periodo: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    ultima_origem: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Resposta (acumulada ou marginal) a cada choque, por horizonte, com bandas HAC.

    `dados` é mensal, indexado por mês, e inclui os meses anteriores a `inicio` usados nas
    defasagens. A amostra de cada h vai de `inicio` a `fim` − h (t + h nunca passa de `fim`).
    Cada contraste (a, b) vira uma linha com choque "a − b" (diferença dos coeficientes, com
    o erro-padrão da mesma matriz HAC). Levanta ValueError se houver faltante na amostra.

    Robustez (Parte 8): `excluir_periodo` = (a, b) tira as origens t cuja janela da dependente
    [t − 1, t + h] cruza [a, b] (os erros HAC tratam a série com o buraco como contínua);
    `ultima_origem` limita t, além de `fim` − h.
    """
    regressores = montar_regressores(dados, espec)
    linhas = []
    for h in espec.horizontes:
        ultimo = fim - pd.DateOffset(months=h)
        if ultima_origem is not None:
            ultimo = min(ultimo, ultima_origem)
        amostra = (dados.index >= inicio) & (dados.index <= ultimo)
        if excluir_periodo is not None:
            a, b = excluir_periodo
            cruza = (dados.index + pd.DateOffset(months=h) >= a) & (
                dados.index - pd.DateOffset(months=1) <= b
            )
            amostra &= ~cruza
        y = dependente(dados, espec.log_preco, h, espec.acumulada)[amostra]
        x = regressores[amostra]
        if y.isna().any() or x.isna().any().any():
            faltam = x.columns[x.isna().any()].tolist() + (["y"] if y.isna().any() else [])
            raise ValueError(f"h={h}: faltantes dentro da amostra em {faltam[:5]}")
        ajuste = sm.OLS(y, x).fit(
            cov_type="HAC", cov_kwds={"maxlags": h + espec.hac_lags_extra, "use_correction": True}
        )
        linhas.extend(_coletar(ajuste, x.columns, espec, h))
    return pd.DataFrame(linhas)


def _coletar(ajuste, colunas: pd.Index, espec: EspecificacaoLP, h: int) -> list[dict]:
    """Linhas da resposta a cada choque e de cada contraste a − b num horizonte."""
    n = int(ajuste.nobs)
    linhas = []
    for choque in espec.choques:
        nome = f"{choque}_l0"
        estimativa = (ajuste.params[nome], ajuste.bse[nome], ajuste.pvalues[nome])
        linhas.append(_linha(choque, h, *estimativa, n, espec.niveis_confianca))
    for a, b in espec.contrastes:
        restricao = (colunas == f"{a}_l0").astype(float) - (colunas == f"{b}_l0")
        teste = ajuste.t_test(restricao)
        estimativa = (float(teste.effect[0]), float(teste.sd[0, 0]), float(teste.pvalue))
        linhas.append(_linha(f"{a} − {b}", h, *estimativa, n, espec.niveis_confianca))
    return linhas


def _meses_desde(meses: pd.Series | pd.Index, origem: pd.Timestamp) -> np.ndarray:
    meses = pd.DatetimeIndex(meses)
    return np.asarray((meses.year - origem.year) * 12 + (meses.month - origem.month))


def estimar_painel(
    dados: pd.DataFrame,
    espec: EspecificacaoLP,
    inicio: pd.Timestamp,
    fim: pd.Timestamp,
    efeito_tempo: bool = False,
    unidade: str = "area_id",
    descartar_incompletos: bool = False,
    ultima_origem: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Local Projection em painel, com erros de Driscoll-Kraay.

    `dados` é longo (colunas `unidade`, mes e as da especificação), mensal e sem lacunas em
    cada unidade, incluindo os meses anteriores a `inicio` usados nas defasagens. As
    defasagens do choque, da inflação e dos controles têm coeficientes comuns às unidades.
    Por unidade entram um efeito fixo, dummies de mês do ano (se `espec.dummies_mes`) e uma
    tendência linear (se `espec.tendencia`). Com `efeito_tempo`, entra também um efeito fixo
    para cada mês t, que absorve tudo o que é comum às unidades naquele mês: o choque precisa
    variar entre unidades, e controles comuns são recusados (seriam colineares). Erros de
    Driscoll-Kraay (soma dos escores por mês + Newey-West com h + hac_lags_extra lags), que
    permitem correlação entre unidades e no tempo. Saída no formato de `estimar`.

    Robustez (Parte 8): `descartar_incompletos` aceita um painel desbalanceado (unidades que
    começam depois de `inicio` entram a partir do 1º mês com todas as defasagens) em vez de
    levantar erro; `ultima_origem` limita t, além de `fim` − h.
    """
    if efeito_tempo and espec.controles:
        raise ValueError("controles comuns às unidades são colineares com o efeito fixo de tempo")
    unidades = sorted(dados[unidade].unique())
    origem = pd.Timestamp(dados["mes"].min())
    sem_deterministicos = replace(espec, dummies_mes=False, tendencia=False)
    series, regressores, deterministicos = {}, [], []
    for u in unidades:
        serie = dados[dados[unidade] == u].set_index("mes").sort_index()
        series[u] = serie
        x = montar_regressores(serie, sem_deterministicos).drop(columns="const")
        regressores.append(x.assign(**{unidade: u}).reset_index())
        det = pd.DataFrame({f"efeito_{u}": 1.0}, index=serie.index)
        # Com efeito fixo de tempo, mês do ano e tendência da 1ª unidade são colineares.
        sem_redundancia = not (efeito_tempo and u == unidades[0])
        if espec.dummies_mes and sem_redundancia:
            for m in range(2, 13):
                det[f"mes{m}_{u}"] = (serie.index.month == m).astype(float)
        if espec.tendencia and sem_redundancia:
            det[f"tendencia_{u}"] = _meses_desde(serie.index, origem) / 12
        deterministicos.append(det.assign(**{unidade: u}).reset_index())
    chaves = [unidade, "mes"]
    empilhado = pd.concat(regressores, ignore_index=True).merge(
        pd.concat(deterministicos, ignore_index=True).fillna(0.0), on=chaves
    )
    linhas = []
    for h in espec.horizontes:
        ultimo = fim - pd.DateOffset(months=h)
        if ultima_origem is not None:
            ultimo = min(ultimo, ultima_origem)
        y = pd.concat(
            [dependente(s, espec.log_preco, h, espec.acumulada).rename("y").to_frame()
             .assign(**{unidade: u}).reset_index() for u, s in series.items()],
            ignore_index=True,
        )  # fmt: skip
        amostra = empilhado.merge(y, on=chaves)
        amostra = amostra[(amostra["mes"] >= inicio) & (amostra["mes"] <= ultimo)]
        if descartar_incompletos:
            amostra = amostra[amostra.drop(columns=chaves).notna().all(axis=1)]
        amostra = amostra.sort_values(["mes", unidade]).reset_index(drop=True)
        x = amostra.drop(columns=[*chaves, "y"])
        if efeito_tempo:
            tempo = pd.get_dummies(amostra["mes"], prefix="t", drop_first=True, dtype=float)
            x = pd.concat([x, tempo], axis=1)
        y_falta = amostra["y"].isna().any()
        if y_falta or x.isna().any().any():
            faltam = x.columns[x.isna().any()].tolist() + (["y"] if y_falta else [])
            raise ValueError(f"h={h}: faltantes dentro da amostra em {faltam[:5]}")
        periodo = _meses_desde(amostra["mes"], inicio)
        ajuste = sm.OLS(amostra["y"], x).fit(
            cov_type="hac-groupsum",
            cov_kwds={"time": periodo, "maxlags": h + espec.hac_lags_extra,
                      "use_correction": "hac"},
        )  # fmt: skip
        linhas.extend(_coletar(ajuste, x.columns, espec, h))
    return pd.DataFrame(linhas)


def desvio_inovacao(choque: pd.Series, lags: int, inicio: pd.Timestamp, fim: pd.Timestamp,
                    dummies_mes: bool = True, tendencia: bool = True) -> float:  # fmt: skip
    """Desvio-padrão da parte do choque em t que suas `lags` defasagens (e as dummies de mês e a
    tendência) não explicam, na amostra de `inicio` a `fim`.

    O coeficiente da LP mantém as defasagens do choque fixas, então mede a resposta a essa
    inovação. Para choques persistentes (SPI acumulado, ONI), a inovação típica é bem menor
    que o desvio-padrão do nível; a razão entre os dois converte a resposta "por 1
    desvio-padrão do nível" em resposta "por 1 inovação típica".
    """
    x = pd.concat({f"l{k}": choque.shift(k) for k in range(1, lags + 1)}, axis=1)
    if dummies_mes:
        meses = pd.get_dummies(choque.index.month, prefix="mes", drop_first=True, dtype=float)
        x = pd.concat([x, meses.set_index(choque.index)], axis=1)
    if tendencia:
        x["tendencia"] = np.arange(len(choque), dtype=float) / 12
    amostra = (choque.index >= inicio) & (choque.index <= fim)
    y, x = choque[amostra], sm.add_constant(x[amostra])
    if y.isna().any() or x.isna().any().any():
        raise ValueError("faltantes dentro da amostra")
    return float(sm.OLS(y, x).fit().resid.std())


def resumo_irf(irf: pd.DataFrame, nivel: int = 90) -> pd.DataFrame:
    """Por especificação e choque: pico (maior |coef|) e seu horizonte, resposta em h = 6, 12 e
    18 e número de horizontes em que a banda de `nivel`% exclui zero (sem correção para testes
    múltiplos: com 19 horizontes correlacionados, alguns excluem zero por acaso)."""
    chaves = [c for c in ["dependente", "especificacao", "choque"] if c in irf.columns]
    linhas = []
    for valores, grupo in irf.groupby(chaves):
        pico = grupo.loc[grupo["coef"].abs().idxmax()]
        exclui = (grupo[f"li_{nivel}"] > 0) | (grupo[f"ls_{nivel}"] < 0)
        por_h = grupo.set_index("horizonte")["coef"]
        linha = dict(zip(chaves, valores, strict=True))
        linha |= {
            "pico_coef": pico["coef"],
            "pico_h": int(pico["horizonte"]),
            "h6": por_h.get(6),
            "h12": por_h.get(12),
            "h18": por_h.get(18),
            f"horizontes_signif_{nivel}": int(exclui.sum()),
            "horizontes": len(grupo),
        }
        linhas.append(linha)
    return pd.DataFrame(linhas)


def avaliar_defasagens(
    acumulada: pd.DataFrame,
    marginal: pd.DataFrame,
    janela: tuple[int, int] | None,
    nivel: int = 90,
    fracao_dissipacao: float = 0.5,
) -> dict:
    """Compara a resposta de um par grupo × choque com a janela de defasagem da hipótese.

    `acumulada` e `marginal` são IRFs de um único choque (colunas horizonte, coef, li_, ls_).
    h_inicio: 1º horizonte com a banda da acumulada acima de zero. lag_otimo: horizonte do
    maior coeficiente da marginal (mês de maior aceleração do preço). Avaliação: compatível
    (a <= h_inicio <= b), antecipada (h_inicio < a), mais lenta (h_inicio > b), contrária (só
    bandas abaixo de zero), inconclusiva (nenhuma banda exclui zero) ou sem hipótese.
    Dissipação: ver `dissipacao`.

    `lag_otimo_banda_acima` NÃO é um teste: o lag ótimo é o maior de vários coeficientes
    ruidosos, então sua banda exclui zero com frequência só por seleção. A evidência sobre a
    marginal é `horizontes_marginal_acima`, a comparar com o esperado por acaso (cerca de
    5% dos horizontes, numa banda de 90%).
    """
    acum = acumulada.sort_values("horizonte").set_index("horizonte")
    marg = marginal.sort_values("horizonte").set_index("horizonte")
    acima = acum.index[acum[f"li_{nivel}"] > 0]
    abaixo = acum.index[acum[f"ls_{nivel}"] < 0]
    h_inicio = int(acima.min()) if len(acima) else None
    if len(acima) == 0:
        avaliacao = "contrária" if len(abaixo) else "inconclusiva"
    elif janela is None:
        avaliacao = "sem hipótese"
    elif h_inicio < janela[0]:
        avaliacao = "antecipada"
    elif h_inicio > janela[1]:
        avaliacao = "mais lenta"
    else:
        avaliacao = "compatível"
    lag_otimo = int(marg["coef"].idxmax())
    return {
        "h_inicio": h_inicio,
        "horizontes_acima": len(acima),
        "horizontes_abaixo": len(abaixo),
        "avaliacao": avaliacao,
        "lag_otimo": lag_otimo,
        "lag_otimo_coef": marg.loc[lag_otimo, "coef"],
        "lag_otimo_banda_acima": bool(marg.loc[lag_otimo, f"li_{nivel}"] > 0),
        "horizontes_marginal_acima": int((marg[f"li_{nivel}"] > 0).sum()),
        **dissipacao(acumulada, nivel, fracao_dissipacao),
    }


def dissipacao(acumulada: pd.DataFrame, nivel: int = 90, fracao: float = 0.5) -> dict:
    """Pico (maior coeficiente) da resposta acumulada e 1º horizonte depois dele em que a
    resposta cai abaixo de `fracao` x pico (None = não cai até o último horizonte).
    `persistencia_h18` = resposta no último horizonte / pico (só para pico positivo)."""
    acum = acumulada.sort_values("horizonte").set_index("horizonte")
    h_pico = int(acum["coef"].idxmax())
    pico = acum.loc[h_pico, "coef"]
    depois = acum.loc[h_pico + 1 :, "coef"]
    caiu = depois.index[depois < fracao * pico]
    return {
        "h_pico": h_pico,
        "pico_coef": pico,
        "pico_signif": bool(acum.loc[h_pico, f"li_{nivel}"] > 0),
        "h_dissipacao": int(caiu.min()) if len(caiu) else None,
        "persistencia_h18": acum["coef"].iloc[-1] / pico if pico > 0 else np.nan,
    }
