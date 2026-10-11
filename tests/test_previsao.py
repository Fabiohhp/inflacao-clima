"""Testes da previsão fora da amostra e dos testes de acurácia, com dados simulados."""

import numpy as np
import pandas as pd
import pytest

from clima_inflacao.modelos import previsao as pv


def _simular(n=360, efeito=0.4, semente=7):
    """Inflação mensal que responde, 1 mês depois, a um preditor persistente; mais um preditor
    de puro ruído."""
    rng = np.random.default_rng(semente)
    meses = pd.date_range("1990-01-01", periods=n, freq="MS")
    sinal = np.zeros(n)
    for t in range(1, n):
        sinal[t] = 0.8 * sinal[t - 1] + rng.normal()
    inflacao = 0.5 + rng.normal(0, 0.5, n)
    inflacao[1:] += efeito * sinal[:-1]
    return pd.DataFrame(
        {"log_p": np.cumsum(inflacao) / 100, "sinal": sinal, "ruido": rng.normal(size=n)},
        index=meses,
    )


def _espec(preditores):
    return pv.EspecificacaoPrevisao(log_preco="log_p", preditores=preditores, lags_inflacao=3,
                                    lags_preditor=2)  # fmt: skip


def test_alvo_e_a_inflacao_dos_meses_seguintes():
    dados = pd.DataFrame({"log_p": np.log([100, 101, 103, 106])},
                         index=pd.date_range("2000-01-01", periods=4, freq="MS"))  # fmt: skip
    y = pv.alvo(dados, "log_p", 2)
    assert y.iloc[0] == pytest.approx(100 * np.log(103 / 100))
    assert y.iloc[-2:].isna().all()


def test_previsao_nao_usa_informacao_posterior_a_origem():
    dados = _simular()
    inicio, origem, fim = dados.index[24], dados.index[200], dados.index[-1]
    base = pv.prever(dados, _espec(["sinal"]), 3, inicio, origem, fim).set_index("origem")
    alterado = dados.copy()
    alterado.loc[alterado.index > origem, ["log_p", "sinal"]] *= 5.0
    nova = pv.prever(alterado, _espec(["sinal"]), 3, inicio, origem, fim).set_index("origem")
    assert nova.loc[origem, "previsao"] == pytest.approx(base.loc[origem, "previsao"])
    # Estimação de `inicio` a origem − h: 200 − 3 − 24 + 1 origens.
    assert base.loc[origem, "n_estimacao"] == 174
    assert base.index.max() == fim - pd.DateOffset(months=3)


def test_janela_movel_fixa_o_tamanho_da_estimacao():
    dados = _simular()
    prev = pv.prever(dados, _espec(["sinal"]), 1, dados.index[24], dados.index[200],
                     dados.index[-1], janela_movel=120)  # fmt: skip
    assert prev["n_estimacao"].eq(120).all()


def test_preditor_verdadeiro_vence_o_benchmark_e_o_ruido_nao():
    dados = _simular()
    args = (1, dados.index[24], dados.index[180], dados.index[-1])
    bench = pv.prever(dados, _espec([]), *args)
    sinal = pv.prever(dados, _espec(["sinal"]), *args)
    ruido = pv.prever(dados, _espec(["ruido"]), *args)
    cw_sinal = pv.clark_west(bench["alvo"], bench["previsao"], sinal["previsao"], 1)
    cw_ruido = pv.clark_west(bench["alvo"], bench["previsao"], ruido["previsao"], 1)
    assert cw_sinal["p_valor"] < 0.01 and pv.r2_fora_amostra(sinal["erro"], bench["erro"]) > 0.2
    assert cw_ruido["p_valor"] > 0.10 and pv.r2_fora_amostra(ruido["erro"], bench["erro"]) < 0.02
    dm = pv.diebold_mariano(ruido["erro"], sinal["erro"], 1)
    assert dm["estatistica"] > 0 and dm["p_valor"] < 0.01  # o ruído erra mais


def test_diebold_mariano_e_simetrico_e_tem_a_correcao_hln():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0, 1.3, 150), rng.normal(0, 1.0, 150)
    ab, ba = pv.diebold_mariano(a, b, 1), pv.diebold_mariano(b, a, 1)
    assert ab["estatistica"] == pytest.approx(-ba["estatistica"])
    assert ab["p_valor"] == pytest.approx(ba["p_valor"])
    d = a**2 - b**2
    sem_correcao = d.mean() / np.sqrt(np.var(d) / 150)
    assert ab["estatistica"] == pytest.approx(sem_correcao * np.sqrt(149 / 150))


def test_variancia_longo_prazo():
    x = np.array([1.0, -1.0, 2.0, 0.0, -2.0])
    assert pv.variancia_longo_prazo(x, 0) == pytest.approx(np.var(x))
    c = x - x.mean()
    esperado = np.var(x) + 2 * 0.5 * (c[1:] @ c[:-1]) / 5
    assert pv.variancia_longo_prazo(x, 1) == pytest.approx(esperado)


def test_r2_fora_amostra_e_r2_defasagens():
    erro = np.array([1.0, -2.0, 0.5])
    assert pv.r2_fora_amostra(erro, erro) == 0
    assert pv.r2_fora_amostra(np.zeros(3), erro) == 1
    dados = _simular()
    ajuste = pv.r2_defasagens(dados["sinal"].shift(-1), dados["sinal"], 2)
    assert ajuste["r2"] > 0.5 and ajuste["r2_ajustado"] < ajuste["r2"]
    assert ajuste["n"] == len(dados) - 3


def test_alvo_cruza_periodo_usa_a_janela_t_mais_1_a_t_mais_h():
    origens = pd.Series(pd.to_datetime(["2019-11-01", "2019-12-01", "2022-12-01", "2021-06-01"]))
    inicio, fim = pd.Timestamp("2020-01-01"), pd.Timestamp("2022-12-01")
    cruza = pv.alvo_cruza_periodo(origens, pd.Series([1, 1, 1, 1]), inicio, fim)
    assert cruza.tolist() == [False, True, False, True]  # alvo de 2019-11 + 1 é 2019-12
    cruza_h12 = pv.alvo_cruza_periodo(origens, pd.Series([12, 12, 12, 12]), inicio, fim)
    assert cruza_h12.tolist() == [True, True, False, True]


def test_avaliar_reproduz_clark_west_e_classifica_diebold_mariano():
    dados = _simular()
    inicio, primeira, fim = (pd.Timestamp(d) for d in ["1991-01-01", "2010-01-01", "2019-12-01"])
    partes = []
    for modelo, preditores in {"benchmark": [], "local": ["ruido"], "mei": ["sinal"]}.items():
        prev = pv.prever(dados, _espec(preditores), 1, inicio, primeira, fim)
        partes.append(prev.assign(dependente="g", modelo=modelo))
    previsoes = pd.concat(partes, ignore_index=True)
    avaliacao, dm = pv.avaliar(previsoes, {"mei": "benchmark", "local": "benchmark"}, ["mei"], 0.10)
    mei = avaliacao.set_index("modelo").loc["mei"]
    por_modelo = {m: g.reset_index(drop=True) for m, g in previsoes.groupby("modelo")}
    cw = pv.clark_west(por_modelo["mei"]["alvo"], por_modelo["benchmark"]["previsao"],
                       por_modelo["mei"]["previsao"], 1)  # fmt: skip
    assert mei["cw_p"] == pytest.approx(cw["p_valor"]) and bool(mei["vence_10pct"])
    assert avaliacao.set_index("modelo").loc["benchmark", "r2_oos_pct"] == 0.0
    assert dm.loc[0, "classe"] == "ENSO superior"  # o preditor verdadeiro erra menos que o ruído
    with pytest.raises(ValueError):
        pv.avaliar(previsoes.iloc[1:], {"mei": "benchmark"}, [], 0.10)
