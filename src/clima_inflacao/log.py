"""Configuração de log comum aos notebooks: console e arquivo em data/logs."""

from __future__ import annotations

import logging
from pathlib import Path


def configurar_log(arquivo: Path | None = None, nivel: int = logging.INFO) -> None:
    """Envia o log para o console e, se `arquivo` for dado, também para esse arquivo."""
    manipuladores: list[logging.Handler] = [logging.StreamHandler()]
    if arquivo is not None:
        arquivo.parent.mkdir(parents=True, exist_ok=True)
        manipuladores.append(logging.FileHandler(arquivo, encoding="utf-8"))
    logging.basicConfig(
        level=nivel,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=manipuladores,
        force=True,
    )
