"""Acesso ao SIDRA/IBGE: metadados de tabelas e consulta de valores."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from clima_inflacao.ingestao.cache import ler_ou_baixar, ler_ou_baixar_json
from clima_inflacao.ingestao.cliente_http import obter_json

URL_METADADOS = "https://servicodados.ibge.gov.br/api/v3/agregados/{tabela}/metadados"
URL_VALORES = "https://apisidra.ibge.gov.br/values"
URL_LOCALIDADES = "https://servicodados.ibge.gov.br/api/v3/agregados/{tabela}/localidades/{nivel}"

# Símbolos do SIDRA na coluna de valor: "-" é zero absoluto (não resultante de arredondamento);
# "..", "..." e "X" são "não se aplica", "não disponível" e "sigiloso".
SIMBOLO_ZERO = "-"
SIMBOLOS_AUSENTES = frozenset({"..", "...", "X"})


def caminho_consulta(
    tabela: int,
    nivel: str,
    *,
    variaveis: Iterable[int],
    periodos: str = "all",
    territorios: str = "all",
    classificacao: int | None = None,
    categorias: Iterable[int | str] | None = None,
) -> str:
    """Monta o caminho da API de valores, ex.: t/7060/n1/all/v/63,66/p/all/c315/7169/d/m."""
    partes = [
        f"t/{tabela}",
        f"{nivel}/{territorios}",
        "v/" + ",".join(map(str, variaveis)),
        f"p/{periodos}",
    ]
    if classificacao is not None:
        categorias = ["all"] if categorias is None else list(categorias)
        partes.append(f"c{classificacao}/" + ",".join(map(str, categorias)))
    partes.append("d/m")  # precisão máxima disponível
    return "/".join(partes)


def metadados(
    tabela: int, *, dir_cache: Path, http: dict[str, Any], refresh: bool = False
) -> dict[str, Any]:
    """Metadados da tabela (períodos, níveis territoriais, variáveis e classificações)."""
    url = URL_METADADOS.format(tabela=tabela)
    return ler_ou_baixar_json(
        dir_cache / f"metadados_t{tabela}.json",
        lambda: obter_json(url, http=http),
        refresh=refresh,
    )


def valores(
    consulta: str, *, dir_cache: Path, http: dict[str, Any], refresh: bool = False
) -> pd.DataFrame:
    """Resposta bruta da API de valores, como texto; a primeira linha é o cabeçalho do SIDRA."""
    nome = consulta.replace("/", "_").replace(",", "-")

    def baixar() -> pd.DataFrame:
        return pd.DataFrame(obter_json(f"{URL_VALORES}/{consulta}", http=http)).astype(str)

    return ler_ou_baixar(dir_cache / f"{nome}.parquet", baixar, refresh=refresh)


def para_numero(valores: pd.Series) -> pd.Series:
    """Converte a coluna de valor do SIDRA em número.

    "-" vira 0 e os símbolos de ausência viram NaN. Qualquer outro texto não numérico gera
    erro, em vez de virar NaN em silêncio.
    """
    texto = valores.astype(str).str.strip()
    texto = texto.mask(texto == SIMBOLO_ZERO, "0").mask(texto.isin(SIMBOLOS_AUSENTES))
    return pd.to_numeric(texto, errors="raise").astype(float)


def arrumar(bruto: pd.DataFrame) -> pd.DataFrame:
    """Converte a resposta bruta em formato longo com nomes de coluna legíveis.

    Supõe a ordem de dimensões de `caminho_consulta`: território (D1), variável (D2),
    período (D3) e, se houver, classificações (D4, D5...). O cabeçalho é conferido.
    """
    cabecalho = bruto.iloc[0]
    dados = bruto.iloc[1:].reset_index(drop=True)
    if not cabecalho["D2N"].startswith("Variável"):
        raise ValueError(f"ordem de dimensões inesperada no SIDRA: {cabecalho.to_dict()}")
    saida = pd.DataFrame(
        {
            "nivel": "n" + dados["NC"],
            "territorio_codigo": dados["D1C"],
            "territorio_nome": dados["D1N"],
            "variavel": dados["D2C"].astype(int),
            "unidade": dados["MN"],
            "periodo": dados["D3C"],
        }
    )
    extras = sorted(
        (c for c in dados.columns if c.startswith("D") and c.endswith("C") and c[1:-1].isdigit()),
        key=lambda c: int(c[1:-1]),
    )[3:]
    for i, coluna in enumerate(extras, start=1):
        sufixo = "" if i == 1 else str(i)
        saida[f"classe{sufixo}_codigo"] = dados[coluna]
        saida[f"classe{sufixo}_nome"] = dados[coluna[:-1] + "N"]
    saida["valor"] = para_numero(dados["V"])
    return saida


def localidades(
    tabela: int, nivel: str, *, dir_cache: Path, http: dict[str, Any], refresh: bool = False
) -> list[dict[str, Any]]:
    """Unidades territoriais de um nível na tabela (atenção: para N7 a API devolve lista vazia)."""
    url = URL_LOCALIDADES.format(tabela=tabela, nivel=nivel.upper())
    return ler_ou_baixar_json(
        dir_cache / f"localidades_t{tabela}_{nivel}.json",
        lambda: obter_json(url, http=http),
        refresh=refresh,
    )


def periodos_da_tabela(meta: dict[str, Any], inicio: int | None = None) -> list[str]:
    """Códigos de período (AAAA ou AAAAMM) do início (ou do ano `inicio`) até o fim da tabela."""
    periodicidade = meta["periodicidade"]
    primeiro, ultimo = periodicidade["inicio"], periodicidade["fim"]
    if periodicidade["frequencia"] == "anual":
        primeiro = max(primeiro, inicio) if inicio is not None else primeiro
        return [str(ano) for ano in range(primeiro, ultimo + 1)]
    if periodicidade["frequencia"] == "mensal":
        if inicio is not None:
            primeiro = max(primeiro, inicio * 100 + 1)
        meses = pd.period_range(
            pd.Period(str(primeiro), "M"), pd.Period(str(ultimo), "M"), freq="M"
        )
        return [mes.strftime("%Y%m") for mes in meses]
    raise ValueError(f"periodicidade não suportada: {periodicidade}")


def periodos_por_consulta(
    n_territorios: int, n_categorias: int, n_variaveis: int, limite: int
) -> int:
    """Quantos períodos cabem numa consulta sem passar do limite de valores da API."""
    por_periodo = n_territorios * n_categorias * n_variaveis
    if por_periodo > limite:
        raise ValueError(f"um único período já tem {por_periodo} valores (limite {limite})")
    return limite // por_periodo


def faixas(periodos: list[str], tamanho: int) -> list[str]:
    """Divide uma lista ordenada e contígua de períodos em faixas "início-fim" de até `tamanho`."""
    blocos = [periodos[i : i + tamanho] for i in range(0, len(periodos), tamanho)]
    return [f"{bloco[0]}-{bloco[-1]}" if len(bloco) > 1 else bloco[0] for bloco in blocos]


def consultar_em_blocos(
    tabela: int,
    nivel: str,
    *,
    variaveis: list[int],
    periodos: list[str],
    classificacao: int,
    categorias: list[int],
    limite: int,
    dir_cache: Path,
    http: dict[str, Any],
    refresh: bool = False,
) -> pd.DataFrame:
    """Consulta dividida em faixas de períodos que respeitam o limite de valores; já arrumada."""
    unidades = localidades(tabela, nivel, dir_cache=dir_cache, http=http, refresh=refresh)
    if not unidades:
        raise ValueError(f"tabela {tabela}: a API não listou localidades do nível {nivel}")
    tamanho = periodos_por_consulta(len(unidades), len(categorias), len(variaveis), limite)
    partes = []
    for faixa in faixas(periodos, tamanho):
        consulta = caminho_consulta(
            tabela,
            nivel,
            variaveis=variaveis,
            periodos=faixa,
            classificacao=classificacao,
            categorias=categorias,
        )
        partes.append(arrumar(valores(consulta, dir_cache=dir_cache, http=http, refresh=refresh)))
    return pd.concat(partes, ignore_index=True)
