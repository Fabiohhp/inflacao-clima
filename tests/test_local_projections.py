"""Testes das Local Projections com dados simulados de resposta conhecida."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from clima_inflacao.modelos import local_projections as lp


def _simular(n=600, efeito=0.3, defasagem=2, semente=42):
    rng = np.random.default_rng(semente)
    meses = pd.date_range("1970-01-01", periods=n, freq="MS")
    choque = rng.normal(size=n)
    sazonal = 0.4 * np.sin(2 * np.pi * meses.month.to_numpy() / 12)
    inflacao = 0.5 + sazonal + rng.normal(0, 0.5, n)  # em %
    inflacao[defasagem:] += efeito * choque[:-defasagem]
    log_p = np.cumsum(inflacao) / 100
    return pd.DataFrame({"log_p": log_p, "choque": choque}, index=meses)


def _espec(**extra):
    return lp.EspecificacaoLP(
        log_preco="log_p", choques=["choque"], horizontes=range(0, 7), **extra
    )


def test_dependente_e_a_variacao_acumulada():
    dados = pd.DataFrame(
        {"log_p": np.log([100, 101, 103, 106])},
        index=pd.date_range("2000-01-01", periods=4, freq="MS"),
    )
    y = lp.dependente(dados, "log_p", 1)
    assert y.iloc[1] == pytest.approx(100 * np.log(103 / 100))
    assert np.isnan(y.iloc[0]) and np.isnan(y.iloc[-1])


def test_lp_recupera_a_resposta_conhecida():
    dados = _simular()
    irf = lp.estimar(dados, _espec(), dados.index[24], dados.index[-1]).set_index("horizonte")
    assert irf.loc[[0, 1], "coef"].abs().max() < 0.1
    assert irf.loc[2:6, "coef"].sub(0.3).abs().max() < 0.12
    assert (irf.loc[2:6, "li_90"] > 0).all()  # efeito detectado com folga
    assert (irf["ls_68"] > irf["li_68"]).all() and (
        irf["ls_90"] - irf["li_90"] > irf["ls_68"] - irf["li_68"]
    ).all()


def test_amostra_termina_em_fim_menos_h():
    dados = _simular()
    irf = lp.estimar(dados, _espec(), dados.index[24], dados.index[-1])
    assert irf["n"].diff().dropna().eq(-1).all()


def test_faltante_dentro_da_amostra_gera_erro():
    dados = _simular()
    dados.iloc[100, dados.columns.get_loc("choque")] = np.nan
    with pytest.raises(ValueError, match="faltantes"):
        lp.estimar(dados, _espec(), dados.index[24], dados.index[-1])


def test_controles_entram_so_defasados_e_sem_tendencia_quando_pedido():
    dados = _simular().assign(log_x=lambda d: np.cumsum(np.ones(len(d))) / 100)
    regressores = lp.montar_regressores(
        dados, _espec(controles=["log_x"], lags_controles=2, tendencia=False)
    )
    assert {"d_log_x_l1", "d_log_x_l2"} <= set(regressores.columns)
    assert "d_log_x_l0" not in regressores.columns and "tendencia" not in regressores.columns
    assert sum(c.startswith("mes_") for c in regressores.columns) == 11


def test_resumo_irf():
    irf = pd.DataFrame(
        {"especificacao": "p", "choque": "c", "horizonte": range(4), "coef": [0.1, -0.5, 0.2, 0.0],
         "li_90": [-0.1, -0.9, 0.05, -0.2], "ls_90": [0.3, -0.1, 0.4, 0.2]}
    )  # fmt: skip
    resumo = lp.resumo_irf(irf).iloc[0]
    assert resumo["pico_h"] == 1 and resumo["pico_coef"] == -0.5
    assert resumo["horizontes_signif_90"] == 2


def test_dependente_marginal_e_a_inflacao_do_mes_t_mais_h():
    dados = pd.DataFrame(
        {"log_p": np.log([100, 101, 103, 106])},
        index=pd.date_range("2000-01-01", periods=4, freq="MS"),
    )
    y = lp.dependente(dados, "log_p", 2, acumulada=False)
    assert y.iloc[1] == pytest.approx(100 * np.log(106 / 103))
    marginal_h0 = lp.dependente(dados, "log_p", 0, acumulada=False)
    assert marginal_h0.equals(lp.dependente(dados, "log_p", 0))


def test_lp_marginal_recupera_o_mes_do_efeito():
    dados = _simular()
    espec = _espec(acumulada=False)
    irf = lp.estimar(dados, espec, dados.index[24], dados.index[-1]).set_index("horizonte")
    assert irf.loc[2, "coef"] == pytest.approx(0.3, abs=0.1) and irf.loc[2, "li_90"] > 0
    assert irf.drop(index=2)["coef"].abs().max() < 0.1  # efeito único na inflação de t+2


def test_contraste_e_a_diferenca_dos_coeficientes():
    dados = _simular()
    rng = np.random.default_rng(1)
    dados["outro"] = rng.normal(size=len(dados))  # sem efeito
    espec = lp.EspecificacaoLP(
        log_preco="log_p", choques=["choque", "outro"], horizontes=range(2, 4),
        contrastes=[("choque", "outro")],
    )  # fmt: skip
    irf = lp.estimar(dados, espec, dados.index[24], dados.index[-1])
    por = irf.set_index(["choque", "horizonte"])["coef"]
    assert por[("choque − outro", 2)] == pytest.approx(por[("choque", 2)] - por[("outro", 2)])
    linha = irf[(irf["choque"] == "choque − outro") & (irf["horizonte"] == 2)].iloc[0]
    assert linha["p_valor"] < 0.01 and linha["ep"] > 0


def _irf(coefs, li, ls=None):
    ls = ls if ls is not None else [c + 1 for c in coefs]
    return pd.DataFrame({"horizonte": range(len(coefs)), "coef": coefs, "li_90": li, "ls_90": ls})


@pytest.mark.parametrize(
    ("li", "janela", "esperado"),
    [
        ([-1, 0.1, 0.2, 0.1, -1], (0, 1), "compatível"),
        ([0.1, 0.1, 0.2, 0.1, -1], (2, 4), "antecipada"),
        ([-1, -1, -1, 0.1, 0.1], (0, 1), "mais lenta"),
        ([-1, -1, -1, -1, -1], (0, 1), "inconclusiva"),
        ([-1, -1, 0.1, -1, -1], None, "sem hipótese"),
    ],
)
def test_avaliar_defasagens_classifica_pela_janela(li, janela, esperado):
    acum = _irf([0.0, 0.5, 1.0, 0.8, 0.2], li)
    marg = _irf([0.0, 0.5, 0.5, -0.2, -0.6], [-1] * 5)
    resultado = lp.avaliar_defasagens(acum, marg, janela)
    assert resultado["avaliacao"] == esperado
    assert resultado["lag_otimo"] == 1  # 1º máximo da marginal
    assert resultado["h_pico"] == 2 and resultado["h_dissipacao"] == 4  # 0,2 < 0,5 x 1,0


def test_avaliar_defasagens_contraria_e_persistente():
    acum = _irf([-0.5, -1.0, -1.2], [-2, -2, -2], ls=[-0.1, -0.2, 0.3])
    resultado = lp.avaliar_defasagens(acum, _irf([0.1, 0.0, 0.0], [-1] * 3), (0, 1))
    assert resultado["avaliacao"] == "contrária" and resultado["h_inicio"] is None
    persistente = lp.avaliar_defasagens(_irf([0.2, 0.9, 1.0], [0.1] * 3), _irf([0.0] * 3, [-1] * 3),
                                        (0, 1))  # fmt: skip
    assert persistente["h_dissipacao"] is None and persistente["persistencia_h18"] == 1.0


def _simular_painel(unidades=8, n=320, efeito=0.3, semente=7):
    """Choque local = 0,8 x choque comum + parte própria; o choque comum também move a
    inflação de todas as unidades no mesmo mês (confundidor). Efeito local real: 0 em h = 0 e
    `efeito` a partir de h = 1."""
    rng = np.random.default_rng(semente)
    meses = pd.date_range("1980-01-01", periods=n, freq="MS")
    comum = rng.normal(size=n)
    partes = []
    for u in range(unidades):
        local = 0.8 * comum + rng.normal(size=n)
        sazonal = 0.3 * np.sin(2 * np.pi * (meses.month.to_numpy() + u) / 12)
        inflacao = 0.4 + sazonal + 1.0 * comum + rng.normal(0, 0.5, n)
        inflacao[1:] += efeito * local[:-1]
        partes.append(pd.DataFrame({"area_id": f"a{u}", "mes": meses, "choque": local,
                                    "log_p": np.cumsum(inflacao) / 100}))  # fmt: skip
    return pd.concat(partes, ignore_index=True)


def test_painel_com_efeito_tempo_remove_o_choque_comum():
    dados = _simular_painel()
    inicio, fim = dados["mes"].min() + pd.DateOffset(months=24), dados["mes"].max()
    espec = lp.EspecificacaoLP(log_preco="log_p", choques=["choque"], horizontes=range(0, 4))
    com = lp.estimar_painel(dados, espec, inicio, fim, efeito_tempo=True).set_index("horizonte")
    sem = lp.estimar_painel(dados, espec, inicio, fim).set_index("horizonte")
    assert abs(com.loc[0, "coef"]) < 0.06 and (com.loc[0, "li_90"] < 0 < com.loc[0, "ls_90"])
    assert com.loc[1:3, "coef"].sub(0.3).abs().max() < 0.08
    assert (com.loc[1:3, "li_90"] > 0).all()
    assert sem.loc[0, "coef"] > 0.3  # sem efeito de tempo, o choque comum contamina h = 0
    assert (com["ls_90"] > com["li_90"]).all()


def test_painel_amostra_e_controles_comuns():
    dados = _simular_painel(unidades=3, n=120)
    inicio, fim = dados["mes"].min() + pd.DateOffset(months=24), dados["mes"].max()
    espec = lp.EspecificacaoLP(log_preco="log_p", choques=["choque"], horizontes=range(0, 3))
    irf = lp.estimar_painel(dados, espec, inicio, fim, efeito_tempo=True)
    assert irf["n"].tolist() == [3 * 96, 3 * 95, 3 * 94]
    dados["log_x"] = np.tile(np.arange(120) / 100, 3)
    with pytest.raises(ValueError, match="colineares"):
        lp.estimar_painel(dados, replace(espec, controles=["log_x"]), inicio, fim, True)


def test_desvio_inovacao_de_um_choque_persistente():
    rng = np.random.default_rng(11)
    meses = pd.date_range("1900-01-01", periods=3000, freq="MS")
    persistente, independente = np.zeros(3000), rng.normal(size=3000)
    for t in range(1, 3000):
        persistente[t] = 0.9 * persistente[t - 1] + rng.normal()
    args = (12, meses[24], meses[-1])
    # Inovação verdadeira com desvio 1; o nível do AR(1) tem desvio 1 / sqrt(1 − 0,81) ≈ 2,29.
    serie = pd.Series(persistente, index=meses)
    assert lp.desvio_inovacao(serie, *args) == pytest.approx(1, abs=0.03)
    assert serie[24:].std() > 2
    ruido = pd.Series(independente, index=meses)
    assert lp.desvio_inovacao(ruido, *args) / ruido[24:].std() == pytest.approx(1, abs=0.03)
