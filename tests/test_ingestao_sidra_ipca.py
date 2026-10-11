"""Testes do parser do SIDRA e do tratamento do IPCA (sem acesso à rede)."""

import numpy as np
import pandas as pd
import pytest

from clima_inflacao.ingestao import ipca, sidra


def test_caminho_consulta():
    caminho = sidra.caminho_consulta(
        7060, "n1", variaveis=[63, 66], classificacao=315, categorias=[7169, 7171]
    )
    assert caminho == "t/7060/n1/all/v/63,66/p/all/c315/7169,7171/d/m"


def test_para_numero_trata_simbolos():
    valores = pd.Series(["0.39", "-0.21", "-", "...", "..", "X", " 1.5 "])
    resultado = sidra.para_numero(valores)
    esperado = [0.39, -0.21, 0.0, np.nan, np.nan, np.nan, 1.5]
    np.testing.assert_array_equal(resultado.to_numpy(), esperado)


def test_para_numero_rejeita_simbolo_desconhecido():
    with pytest.raises(ValueError):
        sidra.para_numero(pd.Series(["1.0", "n/d"]))


def test_arrumar_resposta_bruta():
    cabecalho = {
        "NC": "Nível Territorial (Código)", "NN": "Nível Territorial",
        "MC": "Unidade de Medida (Código)", "MN": "Unidade de Medida", "V": "Valor",
        "D1C": "Brasil (Código)", "D1N": "Brasil",
        "D2C": "Variável (Código)", "D2N": "Variável",
        "D3C": "Mês (Código)", "D3N": "Mês",
        "D4C": "Geral (Código)", "D4N": "Geral",
    }  # fmt: skip
    linha = {
        "NC": "1", "NN": "Brasil", "MC": "2", "MN": "%", "V": "...",
        "D1C": "1", "D1N": "Brasil", "D2C": "63", "D2N": "IPCA - Variação mensal",
        "D3C": "202001", "D3N": "janeiro 2020", "D4C": "7171", "D4N": "11.Alimentação no domicílio",
    }  # fmt: skip
    arrumado = sidra.arrumar(pd.DataFrame([cabecalho, linha]))
    assert arrumado.loc[0, "nivel"] == "n1"
    assert arrumado.loc[0, "variavel"] == 63
    assert arrumado.loc[0, "classe_codigo"] == "7171"
    assert np.isnan(arrumado.loc[0, "valor"])


@pytest.mark.parametrize(
    ("nome", "codigo"),
    [
        ("Índice geral", "geral"),
        ("11.Alimentação no domicílio", "11"),
        ("1111.Leites e derivados", "1111"),
        ("1101001.Arroz", "1101001"),
        ("Sem código", None),
    ],
)
def test_codigo_estrutural(nome, codigo):
    assert ipca.codigo_estrutural(nome) == codigo


def _metadados(*nomes):
    categorias = [{"id": 100 + i, "nome": nome} for i, nome in enumerate(nomes)]
    return {"classificacoes": [{"id": 315, "categorias": categorias}]}


def test_mapear_categorias_usa_codigo_estrutural():
    metas = {
        58: _metadados("Índice geral", "1111.Leite e derivados"),
        655: _metadados("Índice geral", "1111.Leites e derivados"),
    }
    mapa = ipca.mapear_categorias(metas, {"geral": "geral", "leite": "1111"})
    assert len(mapa) == 4
    leite = mapa[mapa["categoria"] == "leite"].set_index("tabela")["nome_sidra"]
    assert leite[58] != leite[655]


@pytest.mark.parametrize(
    "nomes", [("Índice geral",), ("Índice geral", "1111.Leite", "1111.Leite de novo")]
)
def test_mapear_categorias_exige_uma_ocorrencia(nomes):
    with pytest.raises(ValueError):
        ipca.mapear_categorias({1: _metadados(*nomes)}, {"leite": "1111"})


def test_encadear_indice():
    indice = ipca.encadear_indice(pd.Series([np.nan, 1.0, 1.0, -2.0]))
    assert np.isnan(indice[0])
    np.testing.assert_allclose(indice[1:], [101.0, 102.01, 102.01 * 0.98])


def test_encadear_indice_usa_trecho_mais_recente():
    indice = ipca.encadear_indice(pd.Series([1.0, np.nan, 1.0, 1.0, np.nan]))
    assert indice[:2].isna().all()
    np.testing.assert_allclose(indice[2:4], [101.0, 102.01])
    assert np.isnan(indice[4])


def test_adicionar_indice_mes_ausente_nas_linhas_quebra_o_trecho(caplog):
    meses = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-04-01"])  # falta março
    dados = pd.DataFrame(
        {"mes": meses, "area_id": "a", "categoria": "c", "variacao": [1.0, 1.0, 1.0]}
    )
    resultado = ipca.adicionar_indice(dados)
    assert resultado["indice"].iloc[:2].isna().all()
    assert resultado["indice"].iloc[2] == pytest.approx(101.0)
    assert "2020-03" in caplog.text
