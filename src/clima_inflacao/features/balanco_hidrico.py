"""Evapotranspiração de referência (ET0, mm/dia) para o balanço hídrico P − ET0.

Penman-Monteith FAO-56 (Allen et al., 1998) e, como alternativa, Hargreaves-Samani (FAO-56,
eq. 52). Os números de equação nos comentários são os do FAO-56.
"""

from __future__ import annotations

import numpy as np

CONSTANTE_SOLAR = 0.0820  # MJ m-2 min-1
STEFAN_BOLTZMANN = 4.903e-9  # MJ K-4 m-2 dia-1


def radiacao_extraterrestre(lat_graus, dia_do_ano):
    """Ra em MJ m-2 dia-1 (eqs. 21 a 25)."""
    phi = np.radians(lat_graus)
    j = np.asarray(dia_do_ano, dtype=float)
    dr = 1 + 0.033 * np.cos(2 * np.pi * j / 365)
    declinacao = 0.409 * np.sin(2 * np.pi * j / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(declinacao), -1.0, 1.0))
    return (
        24 * 60 / np.pi * CONSTANTE_SOLAR * dr
        * (ws * np.sin(phi) * np.sin(declinacao) + np.cos(phi) * np.cos(declinacao) * np.sin(ws))
    )  # fmt: skip


def pressao_saturacao(t):
    """Pressão de saturação do vapor em kPa (eq. 11)."""
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def et0_penman_monteith(tmax, tmin, umidade_rel, vento_2m, radiacao, lat, dia_do_ano, elevacao,
                        albedo=0.23):  # fmt: skip
    """ET0 diária (mm) por Penman-Monteith FAO-56 (eq. 6), com G = 0.

    `umidade_rel` é a umidade relativa média do dia (%), usada na eq. 19; `radiacao` é a
    radiação solar incidente Rs (MJ m-2 dia-1); `elevacao` em metros.
    """
    tmed = (tmax + tmin) / 2
    delta = 4098 * pressao_saturacao(tmed) / (tmed + 237.3) ** 2  # eq. 13
    pressao = 101.3 * ((293 - 0.0065 * elevacao) / 293) ** 5.26  # eq. 7
    gama = 0.000665 * pressao  # eq. 8
    es = (pressao_saturacao(tmax) + pressao_saturacao(tmin)) / 2  # eq. 12
    ea = umidade_rel / 100 * es  # eq. 19
    ra = radiacao_extraterrestre(lat, dia_do_ano)
    rso = (0.75 + 2e-5 * elevacao) * ra  # eq. 37
    rns = (1 - albedo) * radiacao  # eq. 38
    razao = np.minimum(radiacao / rso, 1.0)  # Rs/Rso <= 1
    rnl = (
        STEFAN_BOLTZMANN * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2
        * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * razao - 0.35)
    )  # eq. 39  # fmt: skip
    rn = rns - rnl  # eq. 40
    numerador = 0.408 * delta * rn + gama * 900 / (tmed + 273) * vento_2m * (es - ea)
    return numerador / (delta + gama * (1 + 0.34 * vento_2m))


def et0_hargreaves(tmax, tmin, lat, dia_do_ano):
    """ET0 diária (mm) por Hargreaves-Samani (eq. 52), com Ra convertida em mm (x 0,408)."""
    tmed = (tmax + tmin) / 2
    amplitude = np.maximum(tmax - tmin, 0)
    ra_mm = 0.408 * radiacao_extraterrestre(lat, dia_do_ano)
    return 0.0023 * (tmed + 17.8) * np.sqrt(amplitude) * ra_mm
