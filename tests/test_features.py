"""Testes das features climáticas: ET0, GDD, SPI, extremos e agregação mensal."""

import numpy as np
import pandas as pd
import pytest

from clima_inflacao.features import agregacao, balanco_hidrico, extremos, gdd, spi

REFERENCIA = (pd.Timestamp("1991-01-01"), pd.Timestamp("2020-12-01"))


# --- ET0 (valores de referência dos exemplos do FAO-56) ------------------------------------


def test_radiacao_extraterrestre_exemplo_8_fao56():
    # 20°S, 3 de setembro (dia 246): Ra = 32,2 MJ m-2 dia-1
    assert balanco_hidrico.radiacao_extraterrestre(-20.0, 246) == pytest.approx(32.2, abs=0.1)


def test_penman_monteith_exemplo_18_fao56():
    # Bruxelas, 6 de julho: Tmax 21,5; Tmin 12,3; ea = 1,409 kPa; u2 = 2,078 m/s;
    # Rs = 22,07 MJ m-2 dia-1; 50°48'N; 100 m -> ET0 = 3,9 mm/dia
    es = (balanco_hidrico.pressao_saturacao(21.5) + balanco_hidrico.pressao_saturacao(12.3)) / 2
    umidade = 100 * 1.409 / es
    et0 = balanco_hidrico.et0_penman_monteith(21.5, 12.3, umidade, 2.078, 22.07, 50.8, 187, 100)
    assert et0 == pytest.approx(3.9, abs=0.05)


def test_hargreaves_positivo_e_cresce_com_a_amplitude():
    menor = balanco_hidrico.et0_hargreaves(30.0, 22.0, -15.0, 15)
    maior = balanco_hidrico.et0_hargreaves(34.0, 18.0, -15.0, 15)
    assert 0 < menor < maior


# --- GDD -------------------------------------------------------------------------------------


def test_gdd_nunca_negativo_e_limitado():
    rng = np.random.default_rng(42)
    tmin = rng.uniform(-10, 35, 10_000)
    tmax = tmin + rng.uniform(0, 20, 10_000)
    valores = gdd.graus_dia(tmax, tmin, 10.0, 30.0)
    assert (valores >= 0).all()
    assert (valores <= 20.0).all()


def test_gdd_valores_conhecidos():
    assert gdd.graus_dia(25.0, 15.0, 10.0, 30.0) == pytest.approx(10.0)
    assert gdd.graus_dia(36.0, 24.0, 10.0, 30.0) == pytest.approx(17.0)  # Tmax cortada em 30
    assert gdd.graus_dia(8.0, 2.0, 10.0, 30.0) == pytest.approx(0.0)
    assert gdd.graus_dia_calor(np.array([28.0, 33.0]), 30.0).tolist() == [0.0, 3.0]


# --- SPI -------------------------------------------------------------------------------------


def _serie_gama(anos=range(1981, 2021), zeros=0.0, semente=42):
    rng = np.random.default_rng(semente)
    meses = pd.date_range(f"{anos[0]}-01-01", f"{anos[-1]}-12-01", freq="MS")
    escala = 50 + 100 * (1 + np.sin(2 * np.pi * meses.month / 12))  # sazonalidade
    chuva = rng.gamma(2.0, escala)
    chuva[rng.random(len(meses)) < zeros] = 0.0
    return pd.Series(chuva, index=meses)


@pytest.mark.parametrize("escala", [1, 3, 6])
def test_spi_tem_media_zero_e_desvio_um_na_referencia(escala):
    valores = spi.spi_serie(_serie_gama(), escala, REFERENCIA)
    na_ref = valores[REFERENCIA[0] : REFERENCIA[1]].dropna()
    assert abs(na_ref.mean()) < 0.1
    assert abs(na_ref.std() - 1) < 0.1


def test_spi_com_zeros_usa_centro_da_massa():
    serie = _serie_gama(zeros=0.3)
    valores = spi.spi_serie(serie, 1, REFERENCIA)
    zeros = serie == 0
    janeiro = serie.index.month == 1
    ref = (serie.index >= REFERENCIA[0]) & (serie.index <= REFERENCIA[1])
    q = (serie[janeiro & ref] == 0).mean()
    esperado = spi.stats.norm.ppf(q / 2)
    assert valores[zeros & janeiro].iloc[0] == pytest.approx(esperado)
    assert (valores[~zeros & janeiro] > esperado).all()


def test_spi_e_monotono_dentro_do_mes_do_ano():
    serie = _serie_gama()
    valores = spi.spi_serie(serie, 1, REFERENCIA)
    julho = serie.index.month == 7
    ordem = np.argsort(serie[julho].to_numpy())
    assert (np.diff(valores[julho].to_numpy()[ordem]) >= 0).all()


def test_spi_sem_positivos_suficientes_fica_ausente():
    serie = _serie_gama()
    serie[serie.index.month == 8] = 0.0
    valores = spi.spi_serie(serie, 1, REFERENCIA)
    assert valores[serie.index.month == 8].isna().all()


# --- Extremos e agregação --------------------------------------------------------------------


def test_maior_sequencia_respeita_os_grupos():
    condicao = pd.Series([1, 1, 0, 1, 1, 1, 1, 1, np.nan, 1])
    grupos = [pd.Series(["a"] * 5 + ["b"] * 5)]
    resultado = extremos.maior_sequencia(condicao, grupos)
    assert resultado.to_dict() == {"a": 2, "b": 3}  # a sequência não atravessa o grupo


def _mensal_duas_celulas(n_anos=40, semente=7):
    rng = np.random.default_rng(semente)
    meses = pd.date_range("1981-01-01", periods=12 * n_anos, freq="MS")
    partes = []
    for celula, constante in [("a", False), ("b", True)]:
        sazonal = 30 + 3 * np.sin(2 * np.pi * meses.month.to_numpy() / 12)
        tmax = sazonal if constante else sazonal + rng.normal(0, 1, len(meses))
        colunas = {"celula_id": celula, "mes": meses, "tmax_media_c": tmax}
        partes.append(pd.DataFrame(colunas | {"spi_3": rng.normal(size=len(meses))}))
    return pd.concat(partes, ignore_index=True)


def test_anomalia_padronizada_na_referencia_e_desvio_zero():
    mensal = _mensal_duas_celulas()
    z = extremos.anomalia_padronizada(mensal, "tmax_media_c", ("1991-01", "2020-12"))
    na_ref = mensal["mes"].between("1991-01-01", "2020-12-01") & (mensal["celula_id"] == "a")
    por_mes = z[na_ref].groupby(mensal.loc[na_ref, "mes"].dt.month)
    assert por_mes.mean().abs().max() < 1e-10
    assert por_mes.std().sub(1).abs().max() < 1e-10
    assert z[mensal["celula_id"] == "b"].isna().all()  # desvio zero na referência: NaN


def test_indicadores_limiar_estritos_e_ausentes():
    mensal = _mensal_duas_celulas().iloc[:6].copy()
    mensal["spi_3"] = [-2.0, -1.5, 0.0, 1.5, 1.6, np.nan]
    ind = extremos.indicadores_limiar(mensal, "spi_3", -1.5, 1.5, 1.5, ("1981-01", "1981-06"))
    assert ind["seca"].tolist()[:5] == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert ind["excesso"].tolist()[:5] == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert ind[["seca", "excesso"]].iloc[5].isna().all()
    assert set(ind["calor"].dropna().unique()) <= {0.0, 1.0}


def test_safra_de_janela_que_cruza_o_ano():
    datas = pd.Series(pd.to_datetime(["2020-12-15", "2021-01-10", "2021-02-28"]))
    assert extremos.safra_da_janela(datas, [12, 1, 2]).tolist() == [2021, 2021, 2021]
    assert extremos.dias_esperados(pd.Index([2021, 2024]), [12, 1, 2]).tolist() == [90, 91]


def _diario_sintetico():
    datas = pd.date_range("2020-01-01", "2020-03-15", freq="D")  # março incompleto
    n = len(datas)
    return pd.DataFrame(
        {
            "celula_id": "c1",
            "data": datas,
            "precipitacao_mm": 2.0,
            "temp_max_c": np.where(np.arange(n) % 10 < 3, 36.0, 30.0),  # 3 dias seguidos > 35
            "temp_min_c": np.where(datas.month == 2, 1.0, 18.0),  # geada em fevereiro
            "umidade_rel_pct": 70.0,
            "vento_2m_ms": 2.0,
            "radiacao_mj_m2": 20.0,
        }
    )


def _cfg_features():
    return {
        "gdd": {"soja": {"t_base_c": 10.0, "t_teto_c": 30.0}},
        "graus_dia_calor": {"limiar_c": 30.0},
        "geada": {"tmin_limite_c": 2.0},
        "calor_extremo": {"tmax_limite_c": 35.0},
        "balanco_hidrico": {"albedo": 0.23},
    }


def test_mensal_agrega_e_descarta_mes_incompleto():
    celulas = pd.DataFrame({"celula_id": ["c1"], "lat_celula": [-15.0], "elevacao_m": [500.0]})
    derivado = agregacao.diarias_derivadas(_diario_sintetico(), celulas, _cfg_features())
    resultado = agregacao.mensal(derivado).set_index("mes")
    janeiro, fevereiro, marco = resultado.index
    assert resultado.loc[janeiro, "precipitacao_mm"] == pytest.approx(62.0)
    assert resultado.loc[janeiro, "max_seq_calor"] == 3
    assert resultado.loc[janeiro, "geada"] == 0 and resultado.loc[fevereiro, "geada"] == 1
    assert resultado.loc[fevereiro, "dias_geada"] == 29
    assert (resultado.loc[janeiro, ["et0_pm_mm", "gdd_10_30"]] > 0).all()
    assert resultado.loc[marco].drop(["celula_id"]).isna().all()  # março incompleto


def test_janelas_criticas_cruza_o_ano_e_descarta_safra_incompleta():
    datas = pd.date_range(
        "2021-01-01", "2022-02-28", freq="D"
    )  # dez/2020 falta: safra 2021 incompleta
    diario = pd.DataFrame(
        {
            "celula_id": "c1",
            "data": datas,
            "precipitacao_mm": 1.0,
            "dia_calor": ((datas.month == 1) & (datas.day <= 4)).astype(float),
            "dia_geada": 0.0,
            "graus_dia_calor": 0.0,
        }
    )
    culturas = {"soja": {"meses_criticos": [12, 1, 2], "meses_geada": []}}
    resultado = extremos.janelas_criticas(diario, culturas).set_index("safra")
    assert resultado.loc[2021, ["dias_calor", "chuva_mm"]].isna().all()
    assert resultado.loc[2022, "dias_calor"] == 4
    assert resultado.loc[2022, "max_seq_calor"] == 4
    assert resultado.loc[2022, "chuva_mm"] == pytest.approx(31 + 31 + 28)
    assert np.isnan(resultado.loc[2022, "dias_geada"])


# --- Validação ---------------------------------------------------------------------------------

from clima_inflacao.features import validacao  # noqa: E402


def test_validar_secas_aplica_o_criterio():
    meses = pd.date_range("2020-01-01", periods=6, freq="MS")
    spi_micro = pd.DataFrame(
        {
            "codigo": ["1"] * 6 + ["2"] * 6,
            "uf": "SP",
            "mes": list(meses) * 2,
            "spi_6": [0, -0.5, -1.2, -0.8, 0.2, 0.1] + [0, -0.5, -2.0, -0.4, 0.2, 0.1],
        }
    )
    eventos = [
        {"evento": "seca", "ufs": ["SP"], "inicio": "2020-02", "fim": "2020-04"},
        {"evento": "fora", "ufs": ["SP"], "inicio": "2020-05", "fim": "2020-06"},
    ]
    resumo, series = validacao.validar_secas(spi_micro, eventos, "spi_6")
    seca, fora = resumo.iloc[0], resumo.iloc[1]
    assert seca["spi_medio_minimo"] == pytest.approx(-1.6)
    assert seca["mes_do_minimo"] == "2020-03"
    assert seca["pct_micro_abaixo_de_1_5_no_minimo"] == pytest.approx(50.0)
    assert bool(seca["passou"]) and not bool(fora["passou"])
    assert len(series) == 12


def test_homogeneidade_detecta_mudanca_de_patamar():
    meses = pd.date_range("2010-01-01", "2025-12-01", freq="MS")
    inmet = pd.DataFrame(
        {"codigo": "A1", "mes": meses, "precipitacao_mm": 100.0, "temperatura_c": 20.0}
    )
    depois = meses >= "2024-01-01"
    features = pd.DataFrame(
        {
            "celula_id": "c1",
            "mes": meses,
            "precipitacao_mm": np.where(depois, 68.0, 80.0),  # viés passa de 0,80 para 0,68
            "tmed_c": np.where(depois, 21.5, 21.0),  # diferença passa de +1,0 para +1,5
        }
    )
    tabela = validacao.homogeneidade_nasa_inmet(
        inmet, features, pd.DataFrame({"codigo": ["A1"], "celula_id": ["c1"]}), base=(2010, 2021)
    ).set_index("semestre")
    assert tabela.loc["2015-S1", "chuva_median"] == pytest.approx(1.0)
    assert tabela.loc["2024-S1", "chuva_median"] == pytest.approx(0.85)
    assert tabela.loc["2024-S2", "temperatura_median"] == pytest.approx(0.5)
