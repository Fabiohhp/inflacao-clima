"""Testes dos modelos dinâmicos com dados simulados de relação de longo prazo conhecida."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.coint_tables import c_sjt
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from clima_inflacao.modelos import dinamicos as dn


def _cointegrado(n=600, beta=0.5, alfa=-0.2, impacto=0.1, semente=5):
    """Grão = passeio aleatório; varejo corrige `alfa` do desvio de varejo − beta x grão."""
    rng = np.random.default_rng(semente)
    datas = pd.date_range("1960-01-01", periods=n, freq="MS")
    grao = np.cumsum(rng.normal(0, 0.05, n))
    varejo = np.zeros(n)
    for t in range(1, n):
        desvio = varejo[t - 1] - beta * grao[t - 1]
        varejo[t] = varejo[t - 1] + alfa * desvio + impacto * (grao[t] - grao[t - 1])
        varejo[t] += rng.normal(0, 0.01)
    return pd.DataFrame({"varejo": varejo, "grao": grao}, index=datas)


@pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning")  # statsmodels
def test_traco_sem_sazonais_reproduz_o_statsmodels():
    dados = _cointegrado()
    teste = dn.johansen_traco(dados, k_ar_diff=2, sazonal=False)
    referencia = coint_johansen(dados.to_numpy(), 0, 2)
    assert teste["autovalor"].to_numpy() == pytest.approx(referencia.eig)
    assert teste["traco"].to_numpy() == pytest.approx(referencia.lr1)
    assert teste["cv_95"].to_numpy() == pytest.approx(referencia.cvt[:, 1])
    assert teste["n"].iloc[0] == len(dados) - 3


@pytest.fixture(scope="module")
def cv_caso2():
    return dn.simular_valores_criticos(2, [1, 2], t=500, replicas=4000, semente=3)


def test_simulador_reproduz_a_tabela_sem_deterministicos():
    simulado = dn.simular_valores_criticos(1, [1, 2], t=500, replicas=4000, semente=3)
    for d in (1, 2):
        assert simulado.loc[d, "cv_95"] == pytest.approx(c_sjt(d, -1)[1], rel=0.05)


def test_pantula_escolhe_posto_um_e_caso_2_sem_deriva(cv_caso2):
    posto, caso, testes = dn.pantula(_cointegrado(), k_ar_diff=1, valores_criticos_caso2=cv_caso2)
    assert (posto, caso) == (1, 2)
    assert set(testes["caso"]) == {2, 3}
    # O caso 3 sozinho supõe deriva e erra o posto nesse sistema sem deriva.
    assert dn.posto(testes[testes["caso"] == 3]) == 2


def test_pantula_sem_cointegracao(cv_caso2):
    rng = np.random.default_rng(9)
    datas = pd.date_range("1960-01-01", periods=600, freq="MS")
    soltos = pd.DataFrame(np.cumsum(rng.normal(size=(600, 2)), axis=0), index=datas,
                          columns=["varejo", "grao"])  # fmt: skip
    assert dn.pantula(soltos, k_ar_diff=1, valores_criticos_caso2=cv_caso2)[0] == 0


def test_vecm_recupera_elasticidade_ajuste_e_exogeneidade():
    r = dn.vecm(_cointegrado(), k_ar_diff=1)
    assert r["elasticidade_grao_li"] < 0.5 < r["elasticidade_grao_ls"]
    assert r["elasticidade_grao"] == pytest.approx(0.5, abs=0.03)
    assert r["alpha_varejo"] == pytest.approx(-0.2, abs=0.05) and r["alpha_varejo_ls"] < 0
    assert r["alpha_grao_li"] < 0 < r["alpha_grao_ls"]  # grão não reage ao desvio
    assert r["meia_vida_meses"] == pytest.approx(np.log(0.5) / np.log(1 + r["alpha_varejo"]))


def test_ardl_recupera_longo_prazo_impacto_e_bounds():
    dados = _cointegrado()
    resumo, modelo = dn.ardl(dados["varejo"], dados[["grao"]], max_lags=4)
    assert resumo["elasticidade_grao"] == pytest.approx(0.5, abs=0.03)
    assert resumo["velocidade"] == pytest.approx(-0.2, abs=0.05)
    assert resumo["n"] == len(dados) - 4  # hold_back fixo
    leitura = dn.leitura_bounds(
        resumo["bounds_f"], resumo["bounds_cv5_i0"], resumo["bounds_cv5_i1"]
    )
    assert leitura == "relação de longo prazo"
    tabela = dn.multiplicadores_simulados(modelo, "grao", horizonte=60, simulacoes=500,
                                          hac_lags=4, nivel=0.9, semente=1)  # fmt: skip
    pontual = tabela.set_index("horizonte")["coef"]
    assert pontual[0] == pytest.approx(0.1, abs=0.03)  # impacto
    assert pontual[60] == pytest.approx(pontual["longo"], abs=1e-3)
    assert (tabela["li"] <= tabela["coef"]).all() and (tabela["coef"] <= tabela["ls"]).all()


def test_multiplicadores_de_um_ardl_simples():
    params = pd.Series({"y.L1": 0.5, "x.L0": 1.0, "x.L1": 0.0})
    m = dn.multiplicadores(params, "y", "x", 3)
    assert m.tolist() == pytest.approx([1.0, 1.5, 1.75, 1.875])


def test_meia_vida_e_dummies():
    assert dn.meia_vida(-0.5) == pytest.approx(1.0)
    assert np.isnan(dn.meia_vida(0.1)) and np.isnan(dn.meia_vida(-1.5))
    dummies = dn.dummies_mes_centradas(pd.date_range("2000-01-01", periods=24, freq="MS"))
    assert dummies.shape == (24, 11) and np.allclose(dummies.sum(axis=0), 0)
    assert dn.leitura_bounds(3.0, 3.5, 4.5) == "sem relação de longo prazo"
    assert dn.leitura_bounds(4.0, 3.5, 4.5) == "inconclusivo"
