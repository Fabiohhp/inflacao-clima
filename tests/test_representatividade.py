"""Testes do teste de representatividade (Paraná × resto do Brasil) e das novas fontes."""

import numpy as np
import pandas as pd
import pytest

from clima_inflacao.ingestao import conab, ipeadata, producao
from clima_inflacao.modelos import representatividade as rp

CRITERIOS = rp.Criterios(0.90, 1.0, 0.10)


def _precos(n=240, semente=3):
    """Índice geral com 0,5% a.m. e um preço nacional com componente real própria (passeio)."""
    rng = np.random.default_rng(semente)
    meses = pd.date_range("2000-01-01", periods=n, freq="MS")
    geral = np.exp(np.cumsum(np.full(n, 0.005)))
    real = np.exp(np.cumsum(rng.normal(0, 0.04, n)))
    return meses, pd.Series(geral, meses), pd.Series(geral * real * 50, meses), rng


def test_mesmo_preco_em_outra_unidade_chega_perto():
    meses, geral, brasil, _ = _precos()
    m = rp.comparar(brasil * 60, brasil, geral, defasagem=12)
    assert m["correlacao_real"] == pytest.approx(1)
    assert m["diferenca_media"] == pytest.approx(0, abs=1e-9)
    assert m["divergencia_acumulada"] == pytest.approx(0, abs=1e-9)
    assert m["inclinacao"] == pytest.approx(1)
    assert m["inflacao_media"] == pytest.approx(6.0)  # 12 x 0,5 ponto de log
    assert rp.avaliar(m, CRITERIOS)["chega_perto"]


@pytest.mark.filterwarnings("ignore::statsmodels.tools.sm_exceptions.SingularMatrixWarning")
def test_deriva_do_pr_aparece_na_divergencia_e_na_tendencia():  # razão = reta exata (ADF singular)
    meses, geral, brasil, _ = _precos()
    anos = np.arange(len(meses)) / 12
    pr = brasil * np.exp(0.02 * anos)  # PR sobe 2 pontos de log por ano a mais
    m = rp.comparar(pr, brasil, geral, defasagem=12)
    assert m["tendencia_razao_ano"] == pytest.approx(2.0)
    assert m["divergencia_acumulada"] == pytest.approx(2.0 * anos[-1])
    assert m["inflacao_acumulada"] == pytest.approx(0.5 * (len(meses) - 1))
    resultado = rp.avaliar(m, CRITERIOS)
    assert resultado["ok_correlacao"] and not resultado["ok_divergencia"]
    assert not resultado["chega_perto"]


def test_inflacao_comum_infla_so_a_correlacao_nominal():
    meses, _, _, rng = _precos()
    # Dois preços com componentes reais independentes e uma inflação comum que oscila muito.
    geral_alta = pd.Series(np.exp(np.cumsum(rng.normal(0.01, 0.03, len(meses)))), meses)
    pr = geral_alta * np.exp(np.cumsum(rng.normal(0, 0.01, len(meses))))
    br = geral_alta * np.exp(np.cumsum(rng.normal(0, 0.01, len(meses))))
    m = rp.comparar(pd.Series(pr, meses), pd.Series(br, meses), geral_alta, defasagem=12)
    assert m["correlacao_nominal"] > 0.5 > m["correlacao_real"]


def test_buraco_na_serie_nao_vira_variacao_de_outro_horizonte():
    meses, geral, brasil, _ = _precos()
    sp = brasil.copy()
    sp.iloc[100:148] = np.nan  # 48 meses faltando, como o milho de SP
    m = rp.comparar(brasil, sp.dropna(), geral, defasagem=12)
    assert m["n_variacoes"] == len(meses) - 12 - (48 + 12)
    assert m["n_adf"] == 100  # maior trecho contínuo: os 100 meses antes do buraco


def test_maior_trecho_continuo():
    serie = pd.Series([1, np.nan, 1, 1, 1, np.nan, 1, 1])
    assert rp.maior_trecho_continuo(serie).index.tolist() == [2, 3, 4]


def test_anual_com_indice_de_anos():
    anos = pd.Index(range(2000, 2024))
    geral = pd.Series(1.06 ** np.arange(24), anos)
    br = geral * pd.Series(np.exp(np.sin(np.arange(24))), anos)
    m = rp.comparar(br * 2, br, geral)
    assert m["n_variacoes"] == 23 and m["inflacao_media"] == pytest.approx(100 * np.log(1.06))


def test_retrato_uf_e_concordancia():
    meses = pd.date_range("2025-10-01", periods=12, freq="MS")
    base = np.exp(np.linspace(0, 0.3, 12) + 0.05 * np.sin(np.arange(12)))
    precos = pd.DataFrame({"PR": 1.1 * base, "SP": base, "MT": 0.9 * base, "GO": base}, meses)
    precos.loc[meses[3], "GO"] = np.nan  # UF incompleta fica fora
    r = rp.retrato_uf(precos)
    # mediana das outras (SP e MT) = média dos logs = base x raiz(0,9)
    assert r["n_ufs"] == 3 and r["razao_nivel_pct"] == pytest.approx(100 * (1.1 / 0.9**0.5 - 1))
    assert r["correlacao_variacoes"] == pytest.approx(1)
    c = rp.concordancia(pd.Series(60 * base, meses), pd.Series(base, meses))
    assert c["razao_cv_pct"] == pytest.approx(0, abs=1e-9) and c["correlacao_variacoes"] == 1


def test_ipeadata_le_valores_mensais():
    bruto = pd.DataFrame({
        "SERCODIGO": "X", "VALDATA": ["2000-02-01T00:00:00-02:00", "2000-01-01T00:00:00-02:00"],
        "VALVALOR": [2.0, None]})  # fmt: skip
    serie = ipeadata.ler_valores(bruto)
    assert serie.index.tolist() == [pd.Timestamp("2000-01-01"), pd.Timestamp("2000-02-01")]
    assert np.isnan(serie.iloc[0]) and serie.iloc[1] == 2.0
    with pytest.raises(ValueError, match="repetidos"):
        ipeadata.ler_valores(pd.concat([bruto, bruto]))
    with pytest.raises(ValueError, match="1º dia"):
        ipeadata.ler_valores(bruto.assign(VALDATA="2000-01-15T00:00:00-02:00"))


def test_conab_le_virgula_decimal_e_marca_repetidas():
    texto = (
        "produto;classificao_produto;id_produto;uf;regiao;ano;mes;dsc_nivel_comercializacao;"
        "valor_produto_kg\n"
        "MILHO   ;EM GRÃOS ;4742;MT ;CO ;2025;10;ATACADO   ;0,85\n"
        "MILHO   ;EM GRÃOS ;4742;MT ;CO ;2025;10;ATACADO   ;1,16\n"
        "MILHO   ;EM GRÃOS ;4742;PR ;SUL;2025;9;ATACADO   ;1,20\n"
    )
    precos = conab.ler_precos_mensais_uf(texto)
    assert precos["valor"].tolist() == [0.85, 1.16, 1.20]
    assert precos["mes"].iloc[2] == pd.Timestamp("2025-09-01")
    assert precos["repetida"].tolist() == [True, True, False]


def test_preco_implicito_do_brasil_sem_pr(monkeypatch):
    def falso(tabela, *, produtos, **_):
        linhas = [
            ("2000", "n1", "1", "Brasil", 1000.0, 100.0),
            ("2000", "n3", "41", "Paraná", 300.0, 20.0),
            ("2000", "n3", "51", "Mato Grosso", 200.0, 40.0),
        ]
        dados = pd.DataFrame(
            [(p, n, c, nome, v, q, prod, tabela) for p, n, c, nome, v, q in linhas
             for prod in produtos],
            columns=["periodo", "nivel", "territorio_codigo", "territorio_nome",
                     "valor_mil_reais", "quantidade", "produto", "tabela"],
        )  # fmt: skip
        unidades = pd.DataFrame({
            "periodo": "2000", "produto": list(produtos) * 2,
            "variavel": ["quantidade"] * len(produtos) + ["valor_mil_reais"] * len(produtos),
            "unidade": ["Toneladas"] * len(produtos) + ["Mil Reais"] * len(produtos)})  # fmt: skip
        return dados, unidades

    monkeypatch.setattr(producao, "_baixar_tabela", falso)
    config = {"precos_atacado": {"representatividade": {"anual": {"anos": [2000, 2000], "ibge": {
        "niveis": ["n1", "n3"], "tabelas": [{"tabela": 1612, "classificacao": 81, "variaveis": {},
                                            "produtos": {"milho": 2711}}]}}}}}  # fmt: skip
    precos = producao.preco_implicito(config).set_index("territorio")["preco"]
    assert precos["brasil"] == pytest.approx(10.0)
    assert precos["pr"] == pytest.approx(15.0)
    assert precos["brasil_sem_pr"] == pytest.approx(700 / 80)


def test_indice_pesos_fixos_ignora_uf_incompleta_e_renormaliza():
    anos = pd.Index(range(2000, 2004))
    precos = pd.DataFrame({
        "51": np.exp(0.10 * np.arange(4)),  # sobe 10 pontos de log por ano
        "52": np.exp(-0.02 * np.arange(4)),
        "11": [1.0, np.nan, 1.0, 1.0],  # incompleta: fica fora
    }, index=anos)  # fmt: skip
    indice = rp.indice_pesos_fixos(precos, pd.Series({"51": 3.0, "52": 1.0, "11": 4.0}))
    assert np.log(indice).diff().dropna().to_numpy() == pytest.approx([0.07] * 3)
    assert indice.iloc[0] == 1 and indice.attrs["ufs"] == ["51", "52"]
    assert indice.attrs["peso_fora"] == pytest.approx(0.5)
