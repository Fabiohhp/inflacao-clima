"""Verifica se as dependências principais do ambiente importam."""

import importlib

import pytest

MODULOS = [
    "pandas", "numpy", "pyarrow", "duckdb", "requests", "tenacity", "sidrapy", "bcb",
    "yaml", "dotenv", "scipy", "statsmodels", "pycwt", "geopandas", "geobr",
    "matplotlib", "jupytext", "clima_inflacao",
]  # fmt: skip


@pytest.mark.parametrize("modulo", MODULOS)
def test_importa(modulo):
    importlib.import_module(modulo)
