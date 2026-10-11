"""Testes da ingestão de clima: grade do NASA POWER, parser do INMET e pontos das malhas."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from clima_inflacao.ingestao import inmet, malhas, nasa_power

CABECALHO_ANTIGO = """REGIÃO:;CO
UF:;DF
ESTAÇÃO:;BRASILIA
CODIGO (WMO):;A001
LATITUDE:;-15,78944444
LONGITUDE:;-47,92583332
ALTITUDE:;1159,54
DATA DE FUNDAÇÃO (YYYY-MM-DD):;2000-05-07
DATA (YYYY-MM-DD);HORA (UTC);PRECIPITAÇÃO TOTAL, HORÁRIO (mm);TEMPERATURA DO AR - BULBO SECO, HORARIA (°C);
2001-01-01;00:00;,8;20,1;
2001-01-01;01:00;-9999;-9999;
"""

CABECALHO_NOVO = """REGIAO:;CO
UF:;DF
ESTACAO:;BRASILIA
CODIGO (WMO):;A001
LATITUDE:;-15,78944444
LONGITUDE:;-47,92583332
ALTITUDE:;1160,96
DATA DE FUNDACAO:;07/05/00
Data;Hora UTC;PRECIPITAÇÃO TOTAL, HORÁRIO (mm);TEMPERATURA DO AR - BULBO SECO, HORARIA (°C);
2024/01/01;0000 UTC;0;22;
2024/01/01;0100 UTC;;21,2;
"""


def test_celulas_da_grade_merra2():
    pontos = pd.DataFrame(
        {"lat": [-12.60, -12.30, -12.80, -0.1], "lon": [-55.70, -55.35, -55.70, 0.1]}
    )
    resultado = nasa_power.celulas(pontos, 0.5, 0.625)
    assert resultado["celula_id"].iloc[0] == resultado["celula_id"].iloc[1] == "-12.50_-55.625"
    assert resultado["celula_id"].iloc[2] == "-13.00_-55.625"
    assert resultado["celula_id"].iloc[3] == "0.00_0.000"  # sem "-0.00"


def test_ler_resposta_confere_codigo_de_ausente():
    resposta = {
        "header": {"fill_value": -999.0},
        "properties": {"parameter": {"T2M": {"20200101": 25.0, "20200102": -999.0}}},
    }
    tabela = nasa_power.ler_resposta(resposta, -999)
    assert tabela["data"].tolist() == ["20200101", "20200102"]
    with pytest.raises(ValueError):
        nasa_power.ler_resposta(resposta, -99.9)


@pytest.mark.parametrize("texto", [CABECALHO_ANTIGO, CABECALHO_NOVO])
def test_ler_csv_inmet_nos_dois_formatos(texto):
    horario = inmet.ler_csv(texto)
    assert horario["datahora_utc"].dt.hour.tolist() == [0, 1]
    assert horario["temperatura_c"].iloc[1] == pytest.approx(21.2) or np.isnan(
        horario["temperatura_c"].iloc[1]
    )
    assert np.isnan(horario["precipitacao_mm"].iloc[1])  # -9999 ou campo vazio
    assert horario["precipitacao_mm"].iloc[0] in (0.8, 0.0)


@pytest.mark.parametrize(
    "texto",
    [
        CABECALHO_ANTIGO,
        CABECALHO_NOVO,
        CABECALHO_NOVO.replace("REGIAO", "REGI?O").replace("ESTACAO", "ESTAC?O"),
    ],
)
def test_ler_metadados_inmet(texto):
    """Inclui o formato de 2019, com acentos gravados como "?"."""
    meta = inmet.ler_metadados(texto)
    assert meta["codigo"] == "A001"
    assert meta["estacao"] == "BRASILIA"
    assert meta["lat_estacao"] == pytest.approx(-15.78944444)


def test_mensal_exige_horas_validas():
    horas = pd.date_range("2020-01-01 03:00", "2020-03-01 02:00", freq="h")  # jan e fev locais
    horario = pd.DataFrame(
        {"codigo": "A001", "datahora_utc": horas, "precipitacao_mm": 0.1, "temperatura_c": 20.0}
    )
    local = horario["datahora_utc"] - pd.Timedelta(hours=3)
    fevereiro = local.dt.month.eq(2) & local.dt.day.le(5)
    horario.loc[fevereiro, "precipitacao_mm"] = np.nan  # ~17% das horas de fevereiro sem chuva
    resultado = inmet.mensal(horario, -3, 90).set_index("mes")
    assert resultado.loc["2020-01-01", "precipitacao_mm"] == pytest.approx(0.1 * 31 * 24)
    assert np.isnan(resultado.loc["2020-02-01", "precipitacao_mm"])
    assert resultado.loc["2020-02-01", "temperatura_c"] == pytest.approx(20.0)


def test_escolher_estacoes_prefere_serie_mais_longa():
    estacoes = pd.DataFrame(
        {
            "codigo": ["perto", "longe"],
            "lat_estacao": [-15.80, -15.95],
            "lon_estacao": [-47.90, -47.90],
        }
    )
    mensal = pd.DataFrame({"codigo": ["perto"] * 2 + ["longe"] * 5, "precipitacao_mm": 1.0})
    pontos = pd.DataFrame({"ponto_id": ["a", "b"], "lat": [-15.79, 10.0], "lon": [-47.90, 10.0]})
    escolhidas = inmet.escolher_estacoes(mensal, estacoes, pontos, 50)
    assert escolhidas["ponto_id"].tolist() == ["a"]  # "b" não tem estação no raio
    assert escolhidas["codigo"].iloc[0] == "longe"


def test_distancia_km():
    # 1° de latitude ~ 111 km
    assert inmet.distancia_km(0, 0, 1, 0) == pytest.approx(111.2, abs=0.1)


def test_pontos_microrregioes_ficam_dentro_do_poligono():
    poligono = Polygon([(-50, -10), (-49, -10), (-49, -9.9), (-49.9, -9.9), (-49.9, -9), (-50, -9)])
    malha = gpd.GeoDataFrame(
        {"code_micro": [11001], "name_micro": ["Teste"], "abbrev_state": ["RO"]},
        geometry=[poligono],
        crs=4674,
    )
    ponto = malhas.pontos_microrregioes(malha).iloc[0]
    assert ponto["ponto_id"] == "micro_11001"
    assert poligono.contains(gpd.points_from_xy([ponto["lon"]], [ponto["lat"]])[0])


def test_mensal_descarta_horas_repetidas():
    horas = pd.date_range("2020-01-01 03:00", "2020-02-01 02:00", freq="h")
    horario = pd.DataFrame(
        {"codigo": "A001", "datahora_utc": horas, "precipitacao_mm": 1.0, "temperatura_c": 20.0}
    )
    resultado = inmet.mensal(pd.concat([horario, horario.iloc[:100]]), -3, 90)
    assert resultado["precipitacao_mm"].iloc[0] == pytest.approx(31 * 24)
    assert resultado["pct_horas_precip"].iloc[0] == pytest.approx(100.0)
