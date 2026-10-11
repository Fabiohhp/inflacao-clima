"""Testes de consistência do config/settings.yaml."""

import pandas as pd
import pytest

from clima_inflacao.config import (
    RAIZ_PROJETO,
    STATUS_VALIDOS,
    caminho_projeto,
    carregar_config,
    itens_com_status,
)


@pytest.fixture(scope="module")
def config():
    return carregar_config()


def test_secoes_obrigatorias(config):
    secoes = {
        "caminhos", "http", "periodos", "ipca", "producao", "macro", "clima",
        "features", "choques", "espacial", "modelos",
    }  # fmt: skip
    assert secoes <= config.keys()


def test_status_validos(config):
    status = list(itens_com_status(config))
    assert status, "nenhuma entrada com status encontrada"
    invalidos = [(chave, valor) for chave, valor in status if valor not in STATUS_VALIDOS]
    assert not invalidos


def test_caminhos_dentro_do_projeto(config):
    for chave in config["caminhos"]:
        assert caminho_projeto(chave, config).is_relative_to(RAIZ_PROJETO)


def test_periodo_referencia_spi_tem_30_anos(config):
    referencia = config["features"]["spi"]["periodo_referencia"]
    inicio = pd.Period(referencia["inicio"], freq="M")
    fim = pd.Period(referencia["fim"], freq="M")
    assert (fim - inicio).n + 1 == 30 * 12


def test_clima_cobre_referencia_do_spi(config):
    clima_inicio = pd.Period(config["periodos"]["clima_inicio"], freq="M")
    referencia_inicio = pd.Period(config["features"]["spi"]["periodo_referencia"]["inicio"], "M")
    assert clima_inicio <= referencia_inicio


def test_limiares_de_choque(config):
    choques = config["choques"]
    assert choques["spi_seca"] < 0 < choques["spi_excesso"]
    assert all(limiar > 0 for limiar in choques["limiares_alternativos_abs"])


def test_raios_do_cinturao_verde(config):
    raios = config["espacial"]["raios_cinturao_verde_km"]
    assert raios == sorted(raios)
    assert all(raio > 0 for raio in raios)


def test_areas_do_ipca(config):
    areas = config["ipca"]["areas"]["lista"]
    ids = [area["id"] for area in areas]
    assert len(ids) == len(set(ids))
    assert ids.count("brasil") == 1
    assert len(areas) - 1 == 16  # 16 áreas de cobertura, além do Brasil
    assert {area["nivel"] for area in areas} <= {"n1", "n6", "n7"}
    for area in areas:
        if area["id"] == "brasil":
            continue
        assert -34 < area["lat"] < 6 and -74 < area["lon"] < -34  # dentro do Brasil


def test_horizontes_local_projections(config):
    lp = config["modelos"]["local_projections"]
    assert 0 <= lp["horizonte_min"] < lp["horizonte_max"]
    assert lp["hac_lags_extra"] >= 1
