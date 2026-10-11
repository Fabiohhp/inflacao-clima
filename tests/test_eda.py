"""Testes das ferramentas de EDA: STL, ordem de integração, correlação cruzada e wavelet."""

import numpy as np
import pandas as pd
import pytest

from clima_inflacao.modelos import eda

MESES = pd.date_range("2000-01-01", periods=287, freq="MS")


def _rng():
    return np.random.default_rng(42)


def test_stl_separa_a_sazonalidade():
    sazonal = 2 * np.sin(2 * np.pi * MESES.month / 12)
    serie = pd.Series(
        0.01 * np.arange(len(MESES)) + sazonal + _rng().normal(0, 0.1, len(MESES)), MESES
    )
    resultado = eda.dessazonalizar_stl(serie)
    assert np.corrcoef(resultado["sazonal"], sazonal)[0, 1] > 0.98
    perfil = resultado["dessazonalizado"].groupby(MESES.month).mean()
    assert perfil.std() < 0.4 * pd.Series(sazonal).groupby(MESES.month).mean().std()


def test_stl_rejeita_lacunas():
    serie = pd.Series(1.0, MESES)
    serie.iloc[10] = np.nan
    with pytest.raises(ValueError):
        eda.dessazonalizar_stl(serie)


@pytest.mark.parametrize(
    ("geradora", "esperado"),
    [(lambda e: e, "I(0)"), (np.cumsum, "I(1)"), (lambda e: np.cumsum(np.cumsum(e)), "I(2)")],
)
def test_ordem_de_integracao(geradora, esperado):
    serie = pd.Series(geradora(_rng().normal(size=len(MESES))), MESES)
    ordem, detalhe = eda.ordem_integracao(serie, "c")
    assert ordem == esperado
    assert detalhe["conclusao"].iloc[-1] == "estacionária"


def test_ccf_prebranqueada_encontra_a_defasagem():
    rng = _rng()
    x = pd.Series(np.convolve(rng.normal(size=300), [1, 0.6, 0.3], "same")[: len(MESES)], MESES)
    y = -0.5 * x.shift(3) + rng.normal(0, 0.5, len(MESES))
    ccf = eda.ccf_prebranqueada(x, y, 12)
    assert ccf.loc[ccf["r"].idxmin(), "defasagem"] == 3
    assert (ccf["banda_bonferroni"] > ccf["banda_pontual"]).all()


def test_ccf_de_series_independentes_fica_dentro_da_banda_de_bonferroni():
    rng = _rng()
    x = pd.Series(rng.normal(size=len(MESES)), MESES)
    y = pd.Series(rng.normal(size=len(MESES)), MESES)
    ccf = eda.ccf_prebranqueada(x, y, 18)
    assert (ccf["r"].abs() < ccf["banda_bonferroni"]).all()


def _ccf(rs):
    return pd.DataFrame(
        {"defasagem": range(len(rs)), "r": rs, "banda_pontual": 0.12, "banda_bonferroni": 0.18}
    )


@pytest.mark.parametrize(
    ("rs", "janela", "esperado"),
    [
        ([0, -0.3, 0, 0], [0, 1], "compatível"),
        ([0, -0.15, 0, -0.3], [0, 1], "parcial"),
        ([0, 0, 0, -0.3], [0, 1], "fora da hipótese"),
        ([0.2, 0, 0, -0.05], [0, 1], "sem sinal"),
        ([0, -0.3, 0, 0], None, "exploratório"),
    ],
)
def test_avaliar_defasagens(rs, janela, esperado):
    assert eda.avaliar_defasagens(_ccf(rs), janela)["avaliacao"] == esperado


def test_coerencia_wavelet_formato():
    rng = _rng()
    x = pd.Series(rng.normal(size=120), MESES[:120])
    y = x + rng.normal(0, 0.5, 120)
    tabela = eda.coerencia_wavelet(x, y, mc=10, semente=42)
    assert set(tabela.columns) == {
        "mes",
        "periodo",
        "coerencia",
        "fase",
        "significativa",
        "dentro_coi",
    }
    assert tabela["coerencia"].between(0, 1.0 + 1e-9).all()
    assert tabela["coerencia"].mean() > 0.5  # séries fortemente relacionadas


def test_leitura_integracao_reconhece_tendencia_e_quebra():
    rng = _rng()
    tendencia = pd.Series(-0.01 * np.arange(len(MESES)) + rng.normal(0, 0.3, len(MESES)), MESES)
    ordem, _ = eda.ordem_integracao(tendencia, "c")
    diag = eda.diagnostico_quebra_tendencia(tendencia)
    assert eda.leitura_integracao(ordem, "c", diag) in {"I(0)", "I(0) em torno de tendência"}

    quebra = pd.Series(
        np.where(np.arange(len(MESES)) < 150, 0.0, 2.0) + rng.normal(0, 0.3, len(MESES)), MESES
    )
    diag = eda.diagnostico_quebra_tendencia(quebra)
    assert diag["za_p"] < 0.05
    assert abs((diag["za_quebra"] - MESES[150]).days) < 120
    assert eda.leitura_integracao("I(1)", "ct", diag) == "I(1)"  # nível com tendência: mantém


def test_leitura_integracao_mantem_passeio_aleatorio():
    passeio = pd.Series(np.cumsum(_rng().normal(size=len(MESES))), MESES)
    ordem, _ = eda.ordem_integracao(passeio, "c")
    assert eda.leitura_integracao(ordem, "c", eda.diagnostico_quebra_tendencia(passeio)) == "I(1)"
