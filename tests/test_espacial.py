"""Testes da ponderação espacial: pesos de produção, CWI e cinturão verde."""

import numpy as np
import pandas as pd
import pytest

from clima_inflacao.espacial import cinturao_verde, cwi, pesos


def _producao():
    linhas = []
    for ano in range(2000, 2006):
        linhas += [
            {"cultura": "soja", "ano": ano, "codigo": "A", "quantidade_t": 100.0 * (ano - 1999)},
            {"cultura": "soja", "ano": ano, "codigo": "B", "quantidade_t": 100.0},
            {"cultura": "soja", "ano": ano, "codigo": "C", "quantidade_t": np.nan},
        ]
    return pd.DataFrame(linhas)


def test_pesos_defasados_somam_um_e_usam_anos_anteriores():
    resultado = pesos.pesos_defasados(_producao(), anos_aplicacao=[2000, 2004])
    assert np.allclose(resultado.groupby(["cultura", "ano"])["peso"].sum(), 1.0)
    p2004 = resultado[resultado["ano"] == 2004].set_index("codigo")["peso"]
    # 2001-2003: A = 200, 300, 400 (média 300); B = 100 -> pesos 0,75 e 0,25; C sem dado -> 0
    assert p2004["A"] == pytest.approx(0.75)
    assert p2004["C"] == 0
    origem = resultado.drop_duplicates("ano").set_index("ano")["origem"]
    assert origem[2000] == "inicio_da_serie" and origem[2004] == "defasado"


def test_pesos_defasados_nao_usam_o_proprio_ano():
    producao = _producao()
    alterada = producao.copy()
    alterada.loc[(alterada["ano"] == 2004) & (alterada["codigo"] == "A"), "quantidade_t"] = 1e9
    original = pesos.pesos_defasados(producao, [2004])
    choque = pesos.pesos_defasados(alterada, [2004])
    pd.testing.assert_frame_equal(original, choque)


def test_pesos_fixos_e_restricao_a_regiao():
    fixos = pesos.pesos_fixos(_producao(), 2000, 2001)  # A média 150, B 100
    assert fixos.set_index("codigo")["peso"]["A"] == pytest.approx(0.6)
    regiao = pesos.restringir(fixos, ["B", "C"])
    assert regiao["peso"].sum() == pytest.approx(1.0)
    assert regiao.set_index("codigo")["peso"]["B"] == pytest.approx(1.0)


def test_cwi_renormaliza_quando_falta_dado():
    meses = pd.to_datetime(["2004-01-01", "2004-02-01"])
    indicador = pd.DataFrame(
        {"codigo": ["A", "B", "A", "B"], "mes": [meses[0]] * 2 + [meses[1]] * 2,
         "spi_3": [-2.0, 1.0, np.nan, 1.0]}
    )  # fmt: skip
    p = pd.DataFrame({"cultura": "soja", "ano": 2004, "codigo": ["A", "B"], "peso": [0.75, 0.25]})
    resultado = cwi.cwi(indicador, p, ["spi_3"]).set_index("mes")
    assert resultado.loc[meses[0], "spi_3"] == pytest.approx(0.75 * -2 + 0.25 * 1)
    assert resultado.loc[meses[1], "spi_3"] == pytest.approx(1.0)  # só B tem dado
    assert resultado.loc[meses[1], "cobertura_spi_3"] == pytest.approx(0.25)


def test_anomalia_de_rendimento_sem_tendencia():
    anos = np.arange(2000, 2011)
    rendimento = pd.Series(1000 * 1.02 ** (anos - 2000), index=anos)  # tendência pura
    rendimento[2005] *= 0.8  # quebra de 20%
    anomalia = cwi.anomalia_rendimento(rendimento, (2000, 2010))
    assert anomalia[2005] < -15
    assert anomalia.drop(2005).abs().max() < 3


def test_celulas_no_raio_e_agregacao():
    sedes = pd.DataFrame({"area_id": ["x"], "lat": [0.0], "lon": [0.0]})
    celulas = pd.DataFrame(
        {
            "celula_id": ["perto", "meio", "longe"],
            "lat_celula": [0.5, 2.0, 5.0],
            "lon_celula": [0.0] * 3,
        }
    )  # 56, 222 e 556 km
    selecao = cinturao_verde.celulas_no_raio(sedes, celulas, [150, 300])
    por_raio = selecao.groupby("raio_km")["celula_id"].apply(sorted).to_dict()
    assert por_raio == {150: ["perto"], 300: ["meio", "perto"]}
    features = pd.DataFrame(
        {
            "celula_id": ["perto", "meio", "longe"],
            "mes": pd.Timestamp("2020-01-01"),
            "spi_3": [1.0, -1.0, 9.0],
        }
    )
    agregado = cinturao_verde.agregar(features, selecao, ["spi_3"]).set_index("raio_km")
    assert agregado.loc[150, "spi_3"] == pytest.approx(1.0)
    assert agregado.loc[300, "spi_3"] == pytest.approx(0.0)
    assert agregado.loc[300, "n_celulas"] == 2


def test_pesos_defasados_ignoram_anos_sem_dado():
    producao = _producao()
    producao.loc[producao["ano"] <= 2001, "quantidade_t"] = np.nan  # série começa em 2002
    resultado = pesos.pesos_defasados(producao, anos_aplicacao=[2001, 2003])
    origem = resultado.drop_duplicates("ano").set_index("ano")["origem"]
    assert origem[2001] == "inicio_da_serie" and origem[2003] == "defasado"
    assert np.allclose(resultado.groupby("ano")["peso"].sum(), 1.0)
