"""Testes das ferramentas de robustez (Parte 8): exclusão de período, placebo e frações."""

import numpy as np
import pandas as pd
import pytest

from clima_inflacao.features import extremos
from clima_inflacao.modelos import local_projections as lp


def _dados(n=240, semente=4):
    rng = np.random.default_rng(semente)
    meses = pd.date_range("2000-01-01", periods=n, freq="MS")
    choque = rng.normal(size=n)
    inflacao = 0.5 + rng.normal(0, 0.5, n)
    inflacao[1:] += 0.3 * choque[:-1]
    return pd.DataFrame({"log_p": np.cumsum(inflacao) / 100, "choque": choque}, index=meses)


def _espec():
    return lp.EspecificacaoLP(log_preco="log_p", choques=["choque"], horizontes=range(0, 4))


def test_excluir_periodo_tira_as_origens_cuja_janela_cruza_o_periodo():
    dados = _dados()
    inicio, fim = dados.index[24], dados.index[-1]
    periodo = (pd.Timestamp("2010-01-01"), pd.Timestamp("2010-12-01"))
    cheio = lp.estimar(dados, _espec(), inicio, fim).set_index("horizonte")["n"]
    sem = lp.estimar(dados, _espec(), inicio, fim, excluir_periodo=periodo).set_index("horizonte")
    # Origens t com [t − 1, t + h] cruzando 2010: de 2010-01 − h até 2011-01 → 13 + h meses.
    for h in range(4):
        assert cheio[h] - sem.loc[h, "n"] == 13 + h


def test_ultima_origem_limita_a_amostra():
    dados = _dados()
    inicio, fim = dados.index[24], dados.index[-1]
    irf = lp.estimar(dados, _espec(), inicio, fim, ultima_origem=dados.index[-13])
    assert (irf["n"] == len(dados) - 24 - 12).all()  # t vai até o 13º mês antes do fim


def test_placebo_com_choque_futuro_nao_tem_efeito_antes_de_ocorrer():
    dados = _dados(n=600)
    dados["placebo"] = dados["choque"].shift(-12)
    espec = lp.EspecificacaoLP(log_preco="log_p", choques=["placebo"], horizontes=range(0, 12))
    irf = lp.estimar(dados, espec, dados.index[24], dados.index[-1],
                     ultima_origem=dados.index[-13])  # fmt: skip
    assert int(((irf["li_90"] > 0) | (irf["ls_90"] < 0)).sum()) <= 2
    real = lp.estimar(dados, _espec(), dados.index[24], dados.index[-1]).set_index("horizonte")
    assert real.loc[1, "li_90"] > 0  # o choque verdadeiro é detectado


def test_fracoes_em_extremo_por_cultura_e_cinturao():
    meses = pd.to_datetime(["2001-01-01", "2001-02-01"])
    indicadores = pd.DataFrame({
        "celula_id": [1, 2, 1, 2], "mes": np.repeat(meses, 2),
        "seca": [1.0, 0.0, 0.0, np.nan], "excesso": [0.0, 1.0, 0.0, 0.0]})  # fmt: skip
    micro = pd.DataFrame({"codigo": ["a", "b"], "celula_id": [1, 2]})
    pesos = pd.DataFrame({"cultura": "milho", "codigo": ["a", "b"], "peso": [0.75, 0.25]})
    selecao = pd.DataFrame({"area_id": "x", "celula_id": [1, 2], "raio_km": 150})
    fr = extremos.fracoes_em_extremo(indicadores, micro, pesos, ["milho"], selecao,
                                     ["seca", "excesso"]).set_index(["motor", "mes"])  # fmt: skip
    assert fr.loc[("milho", meses[0]), "seca"] == pytest.approx(0.75)
    assert fr.loc[("milho", meses[1]), "seca"] == pytest.approx(0.0)  # renormaliza sem a célula 2
    assert fr.loc[("cinturao", meses[0]), "excesso"] == pytest.approx(0.5)


def _painel(unidades=4, n=200, semente=2):
    rng = np.random.default_rng(semente)
    meses = pd.date_range("1990-01-01", periods=n, freq="MS")
    partes = []
    for u in range(unidades):
        choque = rng.normal(size=n)
        inflacao = 0.4 + rng.normal(0, 0.5, n)
        inflacao[1:] += 0.3 * choque[:-1]
        partes.append(pd.DataFrame({"area_id": f"a{u}", "mes": meses, "choque": choque,
                                    "log_p": np.cumsum(inflacao) / 100}))  # fmt: skip
    return pd.concat(partes, ignore_index=True)


def test_painel_desbalanceado_entra_com_defasagens_completas():
    dados = _painel()
    curta = dados[dados["area_id"] == "a3"].iloc[100:]  # unidade que começa no mês 100
    dados = pd.concat([dados[dados["area_id"] != "a3"], curta], ignore_index=True)
    espec = lp.EspecificacaoLP(log_preco="log_p", choques=["choque"], horizontes=range(0, 2))
    inicio, fim = dados["mes"].min() + pd.DateOffset(months=24), dados["mes"].max()
    with pytest.raises(ValueError, match="faltantes"):
        lp.estimar_painel(dados, espec, inicio, fim, efeito_tempo=True)
    irf = lp.estimar_painel(dados, espec, inicio, fim, efeito_tempo=True,
                            descartar_incompletos=True).set_index("horizonte")  # fmt: skip
    # 3 unidades completas (176 meses em h = 0) + a curta a partir do mês 113 (87 meses).
    assert irf.loc[0, "n"] == 3 * 176 + 87
    assert irf.loc[1, "coef"] == pytest.approx(0.3, abs=0.1)


def test_painel_ultima_origem():
    dados = _painel(unidades=3)
    espec = lp.EspecificacaoLP(log_preco="log_p", choques=["choque"], horizontes=range(0, 2))
    inicio, fim = dados["mes"].min() + pd.DateOffset(months=24), dados["mes"].max()
    irf = lp.estimar_painel(dados, espec, inicio, fim, ultima_origem=fim - pd.DateOffset(months=20))
    assert (irf["n"] == 3 * (176 - 20)).all()
