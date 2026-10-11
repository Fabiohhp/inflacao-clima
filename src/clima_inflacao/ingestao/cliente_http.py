"""Cliente HTTP comum a todas as fontes: timeout, retry com backoff exponencial e log."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeVar

import requests
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)

T = TypeVar("T")

STATUS_TRANSITORIOS = frozenset({429, 500, 502, 503, 504})

_contagem = {"requisicoes": 0}


def total_requisicoes() -> int:
    """Número de requisições HTTP feitas neste processo (evidência de uso do cache)."""
    return _contagem["requisicoes"]


def erro_transitorio(exc: BaseException) -> bool:
    """Falhas de rede, respostas 429/5xx e corpo JSON inválido merecem nova tentativa.

    O BCB já devolveu HTTP 200 com corpo vazio numa consulta que, repetida, funcionou.
    """
    if isinstance(exc, requests.ConnectionError | requests.Timeout | requests.JSONDecodeError):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in STATUS_TRANSITORIOS
    return False


def _politica(http: dict[str, Any]) -> Retrying:
    return Retrying(
        stop=stop_after_attempt(http["tentativas_max"]),
        wait=wait_exponential(
            multiplier=http["backoff_inicial_segundos"], max=http["backoff_max_segundos"]
        ),
        retry=retry_if_exception(erro_transitorio),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )


def _get(
    url: str,
    params: dict[str, Any] | None,
    http: dict[str, Any],
    headers: dict[str, str] | None,
    tentativa: int,
) -> requests.Response:
    log.info("GET %s params=%s (tentativa %d)", url, params, tentativa)
    _contagem["requisicoes"] += 1
    resposta = requests.get(url, params=params, headers=headers, timeout=http["timeout_segundos"])
    resposta.raise_for_status()
    return resposta


def obter(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    http: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> requests.Response:
    """GET com timeout e retry; `http` é a seção homônima do settings.yaml.

    Levanta requests.HTTPError se a última tentativa não devolver 2xx.
    """
    for tentativa in _politica(http):
        with tentativa:
            return _get(url, params, http, headers, tentativa.retry_state.attempt_number)
    raise AssertionError("inalcançável: a política de retry relança o último erro")


def obter_json(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    http: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> Any:
    """Como `obter`, mas devolve o JSON decodificado; corpo inválido também é repetido."""
    for tentativa in _politica(http):
        with tentativa:
            numero = tentativa.retry_state.attempt_number
            return _get(url, params, http, headers, numero).json()
    raise AssertionError("inalcançável: a política de retry relança o último erro")


def executar(funcao: Callable[[], T], *, descricao: str, http: dict[str, Any]) -> T:
    """Chamada de biblioteca que acessa a rede por conta própria (ex.: geobr), com retry e log.

    Como a biblioteca não expõe os erros de rede de forma uniforme, qualquer exceção é repetida
    até o limite de tentativas.
    """
    for tentativa in Retrying(
        stop=stop_after_attempt(http["tentativas_max"]),
        wait=wait_exponential(
            multiplier=http["backoff_inicial_segundos"], max=http["backoff_max_segundos"]
        ),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    ):
        with tentativa:
            log.info("%s (tentativa %d)", descricao, tentativa.retry_state.attempt_number)
            _contagem["requisicoes"] += 1
            return funcao()
    raise AssertionError("inalcançável: a política de retry relança o último erro")
