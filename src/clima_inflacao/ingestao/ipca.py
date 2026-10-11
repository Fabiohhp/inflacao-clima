"""IPCA: mapeamento de códigos entre tabelas do SIDRA, download e índices encadeados.

As tabelas do IPCA mudam a cada atualização da POF. Cada categoria é identificada pelo seu
código estrutural (o prefixo do nome no SIDRA, ex.: "1101" em "1101.Cereais, leguminosas e
oleaginosas"), e o código SIDRA de cada tabela é resolvido nos metadados.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from clima_inflacao.config import caminho_projeto
from clima_inflacao.ingestao import sidra

log = logging.getLogger(__name__)

_PADRAO_CODIGO = re.compile(r"^(\d+)\.")
NOMES_VARIAVEIS = {"variacao_mensal": "variacao", "peso_mensal": "peso"}


def codigo_estrutural(nome: str) -> str | None:
    """Código estrutural do IPCA a partir do nome no SIDRA; None se o nome não tiver código."""
    nome = nome.strip()
    if nome.lower() == "índice geral":
        return "geral"
    achado = _PADRAO_CODIGO.match(nome)
    return achado.group(1) if achado else None


def mapear_categorias(
    metadados: dict[int, dict[str, Any]], categorias: dict[str, str]
) -> pd.DataFrame:
    """Tabela explícita categoria × tabela SIDRA → código SIDRA e nome publicado.

    Levanta ValueError se alguma categoria não aparecer exatamente uma vez em alguma tabela.
    """
    linhas = []
    for tabela, meta in metadados.items():
        if len(meta["classificacoes"]) != 1:
            raise ValueError(f"tabela {tabela}: esperava uma classificação")
        classificacao = meta["classificacoes"][0]
        por_codigo: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for categoria in classificacao["categorias"]:
            codigo = codigo_estrutural(categoria["nome"])
            if codigo is not None:
                por_codigo[codigo].append(categoria)
        for nome, codigo in categorias.items():
            achados = por_codigo.get(codigo, [])
            if len(achados) != 1:
                raise ValueError(
                    f"tabela {tabela}: categoria {nome} (código {codigo}) "
                    f"encontrada {len(achados)} vez(es)"
                )
            linhas.append(
                {
                    "categoria": nome,
                    "codigo_ipca": codigo,
                    "tabela": tabela,
                    "classificacao": classificacao["id"],
                    "codigo_sidra": achados[0]["id"],
                    "nome_sidra": achados[0]["nome"],
                }
            )
    return pd.DataFrame(linhas)


def baixar_ipca(
    config: dict[str, Any],
    *,
    refresh: bool = False,
    categorias: dict[str, str] | None = None,
    tabelas: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Baixa variação mensal e peso das tabelas, áreas e categorias do settings.

    `categorias` (nome -> código estrutural) e `tabelas` (códigos SIDRA) substituem as do
    settings, ex.: para o subitem "Óleo diesel", que só existe a partir da tabela 655.
    Devolve (dados, mapeamento). `dados` tem uma linha por mês × área × categoria, com as
    colunas variacao (%), peso (%) e a tabela SIDRA de origem.
    """
    cfg = config["ipca"]
    http = config["http"]
    dir_cache = caminho_projeto("dados_brutos", config) / "sidra"
    categorias = categorias if categorias is not None else cfg["categorias"]
    tabelas = [t for t in cfg["tabelas"] if tabelas is None or t["tabela"] in tabelas]

    metas = {
        t["tabela"]: sidra.metadados(t["tabela"], dir_cache=dir_cache, http=http, refresh=refresh)
        for t in tabelas
    }
    mapeamento = mapear_categorias(metas, categorias)

    partes = []
    for t in tabelas:
        codigos = mapeamento.loc[mapeamento["tabela"] == t["tabela"], "codigo_sidra"]
        for nivel in cfg["niveis_territoriais"]:
            consulta = sidra.caminho_consulta(
                t["tabela"],
                nivel,
                variaveis=t["variaveis"],
                classificacao=t["classificacao"],
                categorias=codigos,
            )
            bruto = sidra.valores(consulta, dir_cache=dir_cache, http=http, refresh=refresh)
            partes.append(sidra.arrumar(bruto).assign(tabela=t["tabela"]))
    longo = pd.concat(partes, ignore_index=True)
    return _organizar(longo, mapeamento, cfg), mapeamento


def _organizar(longo: pd.DataFrame, mapeamento: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    areas = pd.DataFrame(cfg["areas"]["lista"])[["id", "nivel", "codigo_sidra"]]
    areas = areas.rename(columns={"id": "area_id", "codigo_sidra": "territorio_codigo"})
    areas["territorio_codigo"] = areas["territorio_codigo"].astype(str)
    longo = longo.merge(areas, on=["nivel", "territorio_codigo"], how="left", validate="m:1")
    sem_area = longo.loc[longo["area_id"].isna(), "territorio_nome"].unique()
    if len(sem_area):
        raise ValueError(f"territórios do SIDRA ausentes do settings: {list(sem_area)}")

    categorias = mapeamento[["tabela", "codigo_sidra", "categoria"]].copy()
    categorias["classe_codigo"] = categorias["codigo_sidra"].astype(str)
    longo = longo.merge(
        categorias.drop(columns="codigo_sidra"),
        on=["tabela", "classe_codigo"],
        how="left",
        validate="m:1",
    )

    codigos_variaveis = {v["codigo"]: NOMES_VARIAVEIS[k] for k, v in cfg["variaveis"].items()}
    longo["variavel"] = longo["variavel"].map(codigos_variaveis)
    longo["mes"] = pd.to_datetime(longo["periodo"], format="%Y%m")

    # unstack exige índice único: uma duplicata vira erro, não agregação silenciosa.
    dados = (
        longo.set_index(["mes", "area_id", "categoria", "tabela", "variavel"])["valor"]
        .unstack("variavel")
        .reset_index()
    )
    dados.columns.name = None
    if "peso" not in dados:
        dados["peso"] = np.nan
    # Linhas sem variação e sem peso (ex.: "..." antes da entrada da área) são descartadas;
    # adicionar_indice recompõe a grade mensal, então lacunas internas continuam visíveis.
    dados = dados.dropna(subset=["variacao", "peso"], how="all")
    duplicados = dados.duplicated(["mes", "area_id", "categoria"])
    if duplicados.any():
        raise ValueError(f"{duplicados.sum()} meses repetidos entre tabelas")
    colunas = ["mes", "area_id", "categoria", "variacao", "peso", "tabela"]
    return dados[colunas].sort_values(["area_id", "categoria", "mes"]).reset_index(drop=True)


def encadear_indice(variacao: pd.Series) -> pd.Series:
    """Índice encadeado a partir de variações mensais em %, no trecho contíguo mais recente.

    A série deve estar em ordem e sem meses faltando nas linhas. O índice vale 100 no mês
    anterior ao início do trecho que termina na última variação disponível. Níveis de trechos
    separados por uma lacuna não são comparáveis, então os meses antes da última lacuna ficam
    NaN no índice (a variação continua disponível). Nada é interpolado.
    """
    v = variacao.to_numpy(dtype=float)
    indice = np.full(v.shape, np.nan)
    validos = ~np.isnan(v)
    if validos.any():
        fim = len(v) - 1 - int(np.argmax(validos[::-1]))
        lacunas = np.flatnonzero(~validos[: fim + 1])
        inicio = int(lacunas[-1]) + 1 if len(lacunas) else 0
        trecho = v[inicio : fim + 1]
        indice[inicio : fim + 1] = 100 * np.exp(np.cumsum(np.log1p(trecho / 100)))
    return pd.Series(indice, index=variacao.index, name="indice")


def adicionar_indice(dados: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta a coluna `indice` (encadeado por área e categoria, em grade mensal completa).

    Registra no log as séries cujo índice começa depois da primeira variação, por causa de
    uma lacuna.
    """
    meses = pd.date_range(dados["mes"].min(), dados["mes"].max(), freq="MS", name="mes")
    largo = dados.pivot(index="mes", columns=["area_id", "categoria"], values="variacao")
    largo = largo.reindex(meses)
    indices = largo.apply(encadear_indice).rename_axis("mes")
    for serie in largo.columns:
        primeira_variacao = largo[serie].first_valid_index()
        primeiro_indice = indices[serie].first_valid_index()
        if primeira_variacao != primeiro_indice:
            lacunas = largo.loc[primeira_variacao:primeiro_indice, serie]
            lacunas = [f"{m:%Y-%m}" for m in lacunas[lacunas.isna()].index]
            log.warning(
                "índice de %s começa em %s, depois da lacuna %s (variações anteriores mantidas)",
                "/".join(serie), f"{primeiro_indice:%Y-%m}", ", ".join(lacunas),
            )  # fmt: skip
    indices = indices.stack(["area_id", "categoria"]).rename("indice").reset_index()
    return dados.merge(indices, on=["mes", "area_id", "categoria"], how="left", validate="1:1")


def transicoes(
    dados: pd.DataFrame, mapeamento: pd.DataFrame, area_id: str = "brasil"
) -> pd.DataFrame:
    """Quebras de série nas trocas de tabela: nome publicado e peso antes e depois.

    Os pesos só existem a partir da tabela 2938; nas trocas sem peso dos dois lados, as
    colunas de peso ficam vazias.
    """
    serie = dados[dados["area_id"] == area_id]
    primeiros = serie.groupby("tabela")["mes"].min().sort_values()
    nomes = mapeamento.set_index(["tabela", "categoria"])["nome_sidra"]
    pesos = serie.set_index(["mes", "categoria"])["peso"]
    linhas = []
    for anterior, posterior in zip(primeiros.index[:-1], primeiros.index[1:], strict=True):
        mes_novo = primeiros[posterior]
        mes_velho = mes_novo - pd.DateOffset(months=1)
        for categoria in mapeamento["categoria"].unique():
            linhas.append(
                {
                    "mes_transicao": mes_novo,
                    "tabela_anterior": anterior,
                    "tabela_posterior": posterior,
                    "categoria": categoria,
                    "nome_anterior": nomes[(anterior, categoria)],
                    "nome_posterior": nomes[(posterior, categoria)],
                    "nome_mudou": nomes[(anterior, categoria)] != nomes[(posterior, categoria)],
                    "peso_anterior": pesos.get((mes_velho, categoria), np.nan),
                    "peso_posterior": pesos.get((mes_novo, categoria), np.nan),
                }
            )
    resultado = pd.DataFrame(linhas)
    resultado["variacao_peso_pp"] = resultado["peso_posterior"] - resultado["peso_anterior"]
    return resultado
