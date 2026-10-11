"""Preços agropecuários da Conab (arquivos abertos do Portal de Informações).

O arquivo `PrecosMensalUF.txt` é uma janela móvel (em 2026-10-08, de 2025-10 a 2026-09): o
cache guarda a versão do primeiro download, e um `refresh` traz outra janela.
"""

from __future__ import annotations

from io import StringIO
from typing import Any

import pandas as pd

from clima_inflacao.config import caminho_projeto
from clima_inflacao.ingestao.cache import ler_ou_baixar_texto
from clima_inflacao.ingestao.cliente_http import obter

COLUNAS = {
    "produto": "produto",
    "classificao_produto": "classificacao",  # grafia do arquivo
    "uf": "uf",
    "regiao": "regiao",
    "dsc_nivel_comercializacao": "nivel",
    "valor_produto_kg": "valor",
}


def ler_precos_mensais_uf(texto: str) -> pd.DataFrame:
    """Lê o arquivo (`;` como separador, vírgula decimal, campos com espaços à direita)."""
    bruto = pd.read_csv(StringIO(texto), sep=";", dtype=str)
    faltando = (set(COLUNAS) | {"ano", "mes"}) - set(bruto.columns)
    if faltando:
        raise ValueError(f"colunas ausentes no arquivo da Conab: {sorted(faltando)}")
    bruto = bruto.apply(lambda coluna: coluna.str.strip())
    saida = bruto[list(COLUNAS)].rename(columns=COLUNAS)
    decimal = saida["valor"].str.replace(",", ".", regex=False)
    saida["valor"] = pd.to_numeric(decimal, errors="raise")
    saida.insert(0, "mes", pd.to_datetime(bruto["ano"] + "-" + bruto["mes"].str.zfill(2) + "-01"))
    # O arquivo traz algumas linhas repetidas por mês, produto, UF e nível, às vezes com valores
    # diferentes (ex.: milho no atacado em MT). Nada é descartado aqui; quem agrega decide.
    saida["repetida"] = saida.duplicated(["mes", "produto", "classificacao", "uf", "nivel"],
                                         keep=False)  # fmt: skip
    return saida


def precos_mensais_uf(config: dict[str, Any], *, refresh: bool = False) -> pd.DataFrame:
    """Preços mensais por UF, produto e nível de comercialização (texto em latin-1 na origem)."""
    cfg = config["precos_atacado"]["representatividade"]["conab_uf"]
    caminho = caminho_projeto("dados_brutos", config) / "conab" / "PrecosMensalUF.txt"
    texto = ler_ou_baixar_texto(
        caminho,
        lambda: obter(cfg["url"], http=config["http"]).content.decode("latin-1"),
        refresh=refresh,
    )
    return ler_precos_mensais_uf(texto)
