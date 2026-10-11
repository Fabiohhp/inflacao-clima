"""Modelos dinâmicos de preços: teste de Johansen, VECM e ARDL com bounds test.

Convenção: as variáveis entram em log (preço real ou nominal) e a primeira coluna é o preço
no varejo, sobre o qual a relação de longo prazo é normalizada. Elasticidade de longo prazo =
variação percentual do varejo por 1% de alta permanente do regressor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.ardl import ARDL, UECM, ardl_select_order
from statsmodels.tsa.coint_tables import c_sjt
from statsmodels.tsa.vector_ar.vecm import VECM, select_order


def dummies_mes_centradas(datas: pd.DatetimeIndex) -> np.ndarray:
    """11 dummies de mês centradas (indicadora − 1/12), de fevereiro a dezembro."""
    meses = np.asarray(datas.month)
    return np.column_stack([(meses == m) - 1 / 12 for m in range(2, 13)]).astype(float)


def _residuos(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return y - z @ np.linalg.lstsq(z, y, rcond=None)[0]


def johansen_traco(
    dados: pd.DataFrame,
    k_ar_diff: int,
    sazonal: bool = True,
    caso: int = 3,
    valores_criticos: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Teste do traço de Johansen, caso 2 (constante restrita) ou 3 (constante irrestrita).

    Δy(t) = Π y(t−1) + Σ Γ_j Δy(t−j) + [dummies de mês centradas] + erro, com a constante fora
    (caso 3) ou dentro (caso 2) da relação de longo prazo. Na regressão de posto reduzido,
    dummies, Δy defasados e, no caso 3, a constante saem na 1ª etapa. Valores críticos: caso 3
    de Osterwald-Lenum (`c_sjt` do statsmodels); caso 2 em `valores_criticos` (índice = número
    de tendências comuns n − r; colunas cv_90, cv_95, cv_99), de `simular_valores_criticos`.
    As dummies centradas não mudam a distribuição assintótica. A amostra efetiva começa
    k_ar_diff + 1 meses depois do início de `dados`. Uma linha por hipótese nula (posto ≤ r).
    """
    if dados.isna().any().any():
        raise ValueError("Johansen exige dados sem faltantes")
    if caso not in (2, 3):
        raise ValueError("caso deve ser 2 ou 3")
    y = dados.to_numpy(dtype=float)
    dy = np.diff(y, axis=0)
    k = k_ar_diff
    z0 = dy[k:]
    z1 = y[k:-1]
    t = len(z0)
    primeira_etapa = [dy[k - j : len(dy) - j] for j in range(1, k + 1)]
    if sazonal:
        primeira_etapa.append(dummies_mes_centradas(dados.index[k + 1 :]))
    if caso == 3:
        primeira_etapa.append(np.ones((t, 1)))
    else:
        z1 = np.column_stack([z1, np.ones(t)])
    if primeira_etapa:
        z2 = np.column_stack(primeira_etapa)
        r0, r1 = _residuos(z0, z2), _residuos(z1, z2)
    else:
        r0, r1 = z0, z1
    n = y.shape[1]
    autovalores = _autovalores(r0, r1)[:n]
    linhas = []
    for r in range(n):
        traco = -t * np.log(1 - autovalores[r:]).sum()
        if caso == 3:
            cv90, cv95, cv99 = c_sjt(n - r, 0)
        else:
            if valores_criticos is None:
                raise ValueError("o caso 2 precisa de valores críticos simulados")
            cv90, cv95, cv99 = valores_criticos.loc[n - r, ["cv_90", "cv_95", "cv_99"]]
        linhas.append({
            "caso": caso, "posto_nulo": r, "autovalor": autovalores[r], "traco": traco,
            "cv_90": cv90, "cv_95": cv95, "cv_99": cv99, "rejeita_5": traco > cv95, "n": t,
        })  # fmt: skip
    return pd.DataFrame(linhas)


def _autovalores(r0: np.ndarray, r1: np.ndarray) -> np.ndarray:
    """Autovalores (decrescentes) de S11⁻¹ S10 S00⁻¹ S01 da regressão de posto reduzido."""
    t = len(r0)
    s00, s11, s01 = r0.T @ r0 / t, r1.T @ r1 / t, r0.T @ r1 / t
    matriz = np.linalg.solve(s11, s01.T) @ np.linalg.solve(s00, s01)
    return np.clip(np.sort(np.real(np.linalg.eigvals(matriz)))[::-1], 0, 1 - 1e-12)


def simular_valores_criticos(
    caso: int, dimensoes: list[int], *, t: int, replicas: int, semente: int, lote: int = 2000
) -> pd.DataFrame:
    """Quantis 90/95/99% do traço sob H0 (nenhuma cointegração) por simulação.

    Passeios aleatórios gaussianos sem deriva, de `t` passos, sem defasagens; caso 1 = sem
    termos determinísticos, 2 = constante restrita à relação, 3 = constante irrestrita (com
    DGP sem deriva; as tabelas usuais do caso 3 supõem deriva). Índice = dimensão n − r.
    """
    rng = np.random.default_rng(semente)
    linhas = []
    for d in dimensoes:
        tracos = []
        for inicio in range(0, replicas, lote):
            b = min(lote, replicas - inicio)
            choques = rng.standard_normal((b, t + 1, d))
            y = np.cumsum(choques, axis=1)
            r0, r1 = np.diff(y, axis=1), y[:, :-1, :]
            if caso == 2:
                r1 = np.concatenate([r1, np.ones((b, t, 1))], axis=2)
            elif caso == 3:
                r0 = r0 - r0.mean(axis=1, keepdims=True)
                r1 = r1 - r1.mean(axis=1, keepdims=True)
            s00 = np.einsum("bti,btj->bij", r0, r0) / t
            s11 = np.einsum("bti,btj->bij", r1, r1) / t
            s01 = np.einsum("bti,btj->bij", r0, r1) / t
            matriz = np.linalg.solve(s11, np.transpose(s01, (0, 2, 1))) @ np.linalg.solve(s00, s01)
            autovalores = np.sort(np.real(np.linalg.eigvals(matriz)), axis=1)[:, ::-1][:, :d]
            tracos.append(-t * np.log(1 - np.clip(autovalores, 0, 1 - 1e-12)).sum(axis=1))
        tracos = np.concatenate(tracos)
        q = np.quantile(tracos, [0.90, 0.95, 0.99])
        linhas.append({"dimensao": d, "cv_90": q[0], "cv_95": q[1], "cv_99": q[2]})
    return pd.DataFrame(linhas).set_index("dimensao")


def pantula(
    dados: pd.DataFrame, k_ar_diff: int, valores_criticos_caso2: pd.DataFrame,
    sazonal: bool = True,
) -> tuple[int, int | None, pd.DataFrame]:  # fmt: skip
    """Posto e caso determinístico pelo princípio de Pantula: (r = 0, caso 2), (r = 0, caso 3),
    (r = 1, caso 2)...; fica a 1ª combinação não rejeitada a 5%. Se todas forem rejeitadas, o
    posto é n (sistema estacionário) e o caso fica indefinido. Devolve (posto, caso, testes)."""
    testes = pd.concat([
        johansen_traco(dados, k_ar_diff, sazonal, caso=2, valores_criticos=valores_criticos_caso2),
        johansen_traco(dados, k_ar_diff, sazonal, caso=3),
    ], ignore_index=True)  # fmt: skip
    for r in range(dados.shape[1]):
        for caso in (2, 3):
            linha = testes[(testes["posto_nulo"] == r) & (testes["caso"] == caso)].iloc[0]
            if not linha["rejeita_5"]:
                return r, caso, testes
    return dados.shape[1], None, testes


def posto(teste: pd.DataFrame) -> int:
    """Posto estimado num só caso: a 1ª hipótese nula não rejeitada a 5%, em sequência."""
    for linha in teste.itertuples(index=False):
        if not linha.rejeita_5:
            return int(linha.posto_nulo)
    return len(teste)


def selecionar_k_ar_diff(dados: pd.DataFrame, max_k: int, criterio: str = "aic") -> int:
    """Número de defasagens das diferenças do VECM (constante e dummies de mês), por critério de
    informação numa amostra comum (a de `max_k`)."""
    ordem = select_order(dados, maxlags=max_k, deterministic="co", seasons=12)
    return int(ordem.selected_orders[criterio])


def vecm(dados: pd.DataFrame, k_ar_diff: int, deterministico: str = "co",
         nivel: float = 0.90) -> dict[str, float]:  # fmt: skip
    """VECM de posto 1 normalizado no varejo (1ª coluna), com dummies de mês.

    Devolve as elasticidades de longo prazo de cada regressor (−β), as velocidades de ajuste
    (α) de cada equação, com intervalos de `nivel` e p-valores (H0: α = 0; no grão, é o teste
    de exogeneidade fraca), a meia-vida do ajuste do varejo e o 1º coeficiente de curto prazo
    do grão na equação do varejo (Γ_1).
    """
    resultado = VECM(dados, k_ar_diff=k_ar_diff, coint_rank=1, deterministic=deterministico,
                     seasons=12, first_season=dados.index[0].month - 1).fit()  # fmt: skip
    alfa_teste = 1 - nivel
    beta_ic = resultado.conf_int_beta(alpha=alfa_teste)
    alpha_ic = resultado.conf_int_alpha(alpha=alfa_teste)
    nomes = list(dados.columns)
    saida: dict[str, float] = {"n": resultado.nobs, "k_ar_diff": k_ar_diff}
    for i, nome in enumerate(nomes[1:], start=1):
        li, ls = beta_ic[i][0]
        saida[f"elasticidade_{nome}"] = -float(resultado.beta[i, 0])
        saida[f"elasticidade_{nome}_li"], saida[f"elasticidade_{nome}_ls"] = -ls, -li
    for i, nome in enumerate(nomes):
        li, ls = alpha_ic[i][0]
        saida[f"alpha_{nome}"] = float(resultado.alpha[i, 0])
        saida[f"alpha_{nome}_li"], saida[f"alpha_{nome}_ls"] = float(li), float(ls)
        saida[f"alpha_{nome}_p"] = float(resultado.pvalues_alpha[i, 0])
    saida["meia_vida_meses"] = meia_vida(saida[f"alpha_{nomes[0]}"])
    if k_ar_diff >= 1:
        li, ls = resultado.conf_int_gamma(alpha=alfa_teste)[0][1]
        saida["gamma1_grao"] = float(resultado.gamma[0, 1])
        saida["gamma1_grao_li"], saida["gamma1_grao_ls"] = float(li), float(ls)
    saida["brancura_p"] = float(resultado.test_whiteness(nlags=max(24, k_ar_diff + 2)).pvalue)
    return saida


def meia_vida(velocidade: float) -> float:
    """Meses para fechar metade de um desvio do equilíbrio, com ajuste −1 < velocidade < 0."""
    if not -1 < velocidade < 0:
        return np.nan
    return float(np.log(0.5) / np.log(1 + velocidade))


def ardl(y: pd.Series, x: pd.DataFrame, *, max_lags: int, criterio: str = "aic",
         tendencia: str = "c", caso_bounds: int = 3, hac_lags: int = 12, nivel: float = 0.90,
         ordem_minima: int = 1) -> tuple[dict[str, float], ARDL]:  # fmt: skip
    """ARDL com dummies de mês, ordens por critério de informação e bounds test (PSS, 2001).

    A amostra efetiva começa `max_lags` meses depois do início dos dados (hold_back fixo, igual
    na seleção e na estimação). Ordens abaixo de `ordem_minima` sobem para ela, porque a forma
    de correção de erros precisa do nível defasado de cada variável. O bounds test usa erros
    clássicos; a elasticidade de longo prazo (método delta) e a velocidade de ajuste usam
    erros HAC com `hac_lags`. Devolve (resumo, modelo ARDL ajustável).
    """
    selecao = ardl_select_order(y, max_lags, x, max_lags, trend=tendencia, seasonal=True,
                                ic=criterio, hold_back=max_lags, period=12)  # fmt: skip
    p = max(max(selecao.model.ar_lags or [0]), ordem_minima)
    ordens = {c: max(max(selecao.model.dl_lags.get(c, [0])), ordem_minima) for c in x.columns}
    modelo = ARDL(y, p, x, ordens, trend=tendencia, seasonal=True, hold_back=max_lags,
                  period=12)  # fmt: skip
    erros = {"cov_type": "HAC", "cov_kwds": {"maxlags": hac_lags}}
    uecm = UECM.from_ardl(modelo).fit(**erros)
    bounds = uecm.bounds_test(case=caso_bounds)
    z = stats.norm.ppf(0.5 + nivel / 2)
    nome_y = y.name
    saida: dict[str, float] = {
        "n": int(uecm.nobs), "p": p, **{f"q_{c}": q for c, q in ordens.items()},
        "bounds_f": float(bounds.statistic),
        "bounds_cv5_i0": float(bounds.critical_values.loc[95.0, "lower"]),
        "bounds_cv5_i1": float(bounds.critical_values.loc[95.0, "upper"]),
        "bounds_p_i0": float(bounds.pvalue["lower"]),
        "bounds_p_i1": float(bounds.pvalue["upper"]),
    }  # fmt: skip
    for c in x.columns:
        coef, ep = -float(uecm.ci_params[c]), float(uecm.ci_bse[c])
        saida[f"elasticidade_{c}"] = coef
        saida[f"elasticidade_{c}_li"], saida[f"elasticidade_{c}_ls"] = coef - z * ep, coef + z * ep
    velocidade = float(uecm.params[f"{nome_y}.L1"])
    ep = float(uecm.bse[f"{nome_y}.L1"])
    saida["velocidade"] = velocidade
    saida["velocidade_li"], saida["velocidade_ls"] = velocidade - z * ep, velocidade + z * ep
    saida["meia_vida_meses"] = meia_vida(velocidade)
    return saida, modelo


def leitura_bounds(f: float, cv_i0: float, cv_i1: float) -> str:
    """Leitura do bounds test: acima do limite I(1) há relação de longo prazo."""
    if f > cv_i1:
        return "relação de longo prazo"
    if f < cv_i0:
        return "sem relação de longo prazo"
    return "inconclusivo"


def multiplicadores(params: pd.Series, nome_y: str, nome_x: str, horizonte: int) -> np.ndarray:
    """Multiplicadores dinâmicos acumulados de um ARDL em nível: resposta de y em t + h a uma
    alta permanente de 1 unidade em x a partir de t (h = 0..horizonte)."""
    phi = [params[i] for i in params.index if i.startswith(f"{nome_y}.L")]
    theta = {int(i.rsplit("L", 1)[1]): params[i] for i in params.index
             if i.startswith(f"{nome_x}.L")}  # fmt: skip
    m = np.zeros(horizonte + 1)
    for h in range(horizonte + 1):
        m[h] = sum(phi[i] * m[h - 1 - i] for i in range(len(phi)) if h - 1 - i >= 0)
        m[h] += sum(coef for j, coef in theta.items() if j <= h)
    return m


def multiplicadores_simulados(
    modelo: ARDL,
    nome_x: str,
    *,
    horizonte: int,
    simulacoes: int,
    hac_lags: int,
    nivel: float,
    semente: int,
) -> pd.DataFrame:
    """Multiplicadores acumulados com bandas por sorteio dos parâmetros (normal com a
    covariância HAC), semente fixa. Inclui o multiplicador de longo prazo (h = "longo")."""
    ajuste = modelo.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    params, cov = ajuste.params, ajuste.cov_params()
    nome_y = modelo.endog_names if isinstance(modelo.endog_names, str) else modelo.endog_names[0]
    rng = np.random.default_rng(semente)
    sorteios = rng.multivariate_normal(params.to_numpy(), cov.to_numpy(), size=simulacoes)
    caminhos = np.array([
        multiplicadores(pd.Series(s, index=params.index), nome_y, nome_x, horizonte)
        for s in sorteios
    ])  # fmt: skip
    phi = sorteios[:, [i for i, n in enumerate(params.index) if n.startswith(f"{nome_y}.L")]]
    theta = sorteios[:, [i for i, n in enumerate(params.index) if n.startswith(f"{nome_x}.L")]]
    longo = theta.sum(axis=1) / (1 - phi.sum(axis=1))
    pontual = multiplicadores(params, nome_y, nome_x, horizonte)
    phi0 = params[[n for n in params.index if n.startswith(f"{nome_y}.L")]].sum()
    theta0 = params[[n for n in params.index if n.startswith(f"{nome_x}.L")]].sum()
    a = (1 - nivel) / 2
    tabela = pd.DataFrame({
        "horizonte": list(range(horizonte + 1)) + ["longo"],
        "coef": list(pontual) + [theta0 / (1 - phi0)],
        "li": list(np.quantile(caminhos, a, axis=0)) + [np.quantile(longo, a)],
        "ls": list(np.quantile(caminhos, 1 - a, axis=0)) + [np.quantile(longo, 1 - a)],
    })  # fmt: skip
    return tabela
