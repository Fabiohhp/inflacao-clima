"""Leitura da configuração central (config/settings.yaml) e dos caminhos do projeto."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

RAIZ_PROJETO = Path(__file__).resolve().parents[2]
CAMINHO_CONFIG_PADRAO = RAIZ_PROJETO / "config" / "settings.yaml"

STATUS_VALIDOS = frozenset({"confirmado", "a_confirmar", "a_definir"})

# Entradas de `caminhos` que são diretórios (as demais são arquivos).
_DIRETORIOS = (
    "dados_brutos", "dados_processados", "logs", "figuras", "tabelas",
)  # fmt: skip


def carregar_config(caminho: str | Path | None = None) -> dict[str, Any]:
    """Lê o settings.yaml e devolve a configuração como dicionário."""
    caminho = Path(caminho) if caminho is not None else CAMINHO_CONFIG_PADRAO
    with caminho.open(encoding="utf-8") as arquivo:
        return yaml.safe_load(arquivo)


def caminho_projeto(chave: str, config: dict[str, Any] | None = None) -> Path:
    """Caminho absoluto de uma entrada da seção `caminhos` da configuração."""
    config = config if config is not None else carregar_config()
    return RAIZ_PROJETO / config["caminhos"][chave]


def garantir_diretorios(config: dict[str, Any] | None = None) -> None:
    """Cria os diretórios de dados e de saída, que não são versionados ou podem estar vazios."""
    config = config if config is not None else carregar_config()
    for chave in _DIRETORIOS:
        caminho_projeto(chave, config).mkdir(parents=True, exist_ok=True)


def itens_com_status(config: Any, prefixo: str = "") -> Iterator[tuple[str, str]]:
    """Percorre a configuração e devolve (caminho da chave, status) de cada entrada com `status`."""
    if isinstance(config, dict):
        if "status" in config:
            yield prefixo, config["status"]
        for chave, valor in config.items():
            yield from itens_com_status(valor, f"{prefixo}.{chave}" if prefixo else str(chave))
    elif isinstance(config, list):
        for i, valor in enumerate(config):
            yield from itens_com_status(valor, f"{prefixo}[{i}]")
