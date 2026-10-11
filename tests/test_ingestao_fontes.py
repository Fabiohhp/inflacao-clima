"""Testes dos auxiliares do SIDRA em blocos, do SGS/BCB e do parser da NOAA PSL."""

import numpy as np
import pandas as pd
import pytest

from clima_inflacao.ingestao import bcb, enso, sidra

ONI_EXEMPLO = """ 2025         2026
 2025  -0.45  -0.24  -0.06   0.02  -0.02  -0.04  -0.11  -0.26  -0.43  -0.57  -0.61  -0.60
 2026  -0.39  -0.21   0.11   0.46   0.95   1.39   1.80 -99.90 -99.90 -99.90 -99.90 -99.90
  -99.9
 ONI from CPC
 As of 08/4/2026, uses NOAA ERSST V6
 2027
"""


def test_periodos_da_tabela_anual_e_mensal():
    anual = {"periodicidade": {"frequencia": "anual", "inicio": 1974, "fim": 2025}}
    assert sidra.periodos_da_tabela(anual, 2023) == ["2023", "2024", "2025"]
    mensal = {"periodicidade": {"frequencia": "mensal", "inicio": 200609, "fim": 200702}}
    esperado = ["200609", "200610", "200611", "200612", "200701", "200702"]
    assert sidra.periodos_da_tabela(mensal) == esperado
    assert sidra.periodos_da_tabela(mensal, 2007) == ["200701", "200702"]


def test_periodos_por_consulta_respeita_o_limite():
    # 558 microrregiões × 8 produtos × 3 variáveis = 13.392 valores por ano
    assert sidra.periodos_por_consulta(558, 8, 3, 50_000) == 3
    with pytest.raises(ValueError):
        sidra.periodos_por_consulta(558, 100, 3, 50_000)


def test_faixas():
    periodos = [str(a) for a in range(1990, 1997)]
    assert sidra.faixas(periodos, 3) == ["1990-1992", "1993-1995", "1996"]


def test_janelas_tem_limites_fixos_e_cobrem_o_fim():
    resultado = bcb.janelas(pd.Timestamp("1994-07-01"), pd.Timestamp("2026-10-04"), 10)
    assert resultado[0] == (pd.Timestamp("1994-07-01"), pd.Timestamp("2004-06-30"))
    assert resultado[1][0] == pd.Timestamp("2004-07-01")
    assert resultado[-1][0] <= pd.Timestamp("2026-10-04") <= resultado[-1][1]
    assert len(resultado) == 4


def test_media_mensal():
    diaria = pd.DataFrame(
        {
            "data": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-02-03"]),
            "valor": [4.0, 5.0, 6.0],
        }
    )
    mensal = bcb.media_mensal(diaria)
    assert mensal["mes"].tolist() == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")]
    assert mensal["valor"].tolist() == [4.5, 6.0]
    assert mensal["n_dias"].tolist() == [2, 1]


def test_ler_psl_ignora_rodape_e_marca_ausentes():
    serie, ausente = enso.ler_psl(ONI_EXEMPLO)
    assert ausente == pytest.approx(-99.9)
    assert len(serie) == 24
    assert serie["mes"].iloc[0] == pd.Timestamp("2025-01-01")
    valores = serie.set_index("mes")["valor"]
    assert valores[pd.Timestamp("2026-07-01")] == pytest.approx(1.80)
    assert valores[pd.Timestamp("2026-08-01") :].isna().all()
    assert not np.isclose(valores.dropna(), -99.9).any()


def test_ler_psl_rejeita_formato_inesperado():
    sem_ausente = "\n".join(ONI_EXEMPLO.splitlines()[:3])
    with pytest.raises(ValueError, match="ausente"):
        enso.ler_psl(sem_ausente)
    ano_faltando = ONI_EXEMPLO.replace(" 2025         2026", " 2024         2026")
    with pytest.raises(ValueError, match="anos"):
        enso.ler_psl(ano_faltando)
