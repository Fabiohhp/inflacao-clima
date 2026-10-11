"""Produção agrícola do IBGE: PAM (anual, até microrregião) e LSPA (mensal, até UF)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from clima_inflacao.config import caminho_projeto
from clima_inflacao.ingestao import sidra

log = logging.getLogger(__name__)

_INDICE = ["periodo", "nivel", "territorio_codigo", "produto", "tabela"]


def _baixar_tabela(
    tabela: int,
    *,
    classificacao: int,
    variaveis: dict[str, int],
    produtos: dict[str, int],
    niveis: list[str],
    inicio: int | None,
    config: dict[str, Any],
    refresh: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Baixa uma tabela de produção em blocos.

    Devolve (dados em formato largo, com uma coluna por variável; unidades publicadas).
    """
    http = config["http"]
    dir_cache = caminho_projeto("dados_brutos", config) / "sidra"
    meta = sidra.metadados(tabela, dir_cache=dir_cache, http=http, refresh=refresh)
    periodos = sidra.periodos_da_tabela(meta, inicio)
    longo = pd.concat(
        [
            sidra.consultar_em_blocos(
                tabela,
                nivel,
                variaveis=list(variaveis.values()),
                periodos=periodos,
                classificacao=classificacao,
                categorias=list(produtos.values()),
                limite=config["sidra"]["limite_valores"],
                dir_cache=dir_cache,
                http=http,
                refresh=refresh,
            )
            for nivel in niveis
        ],
        ignore_index=True,
    )
    longo["variavel"] = longo["variavel"].map({v: k for k, v in variaveis.items()})
    longo["produto"] = longo["classe_codigo"].map({str(v): k for k, v in produtos.items()})
    if longo[["variavel", "produto"]].isna().any().any():
        raise ValueError(f"tabela {tabela}: variável ou produto fora do settings na resposta")
    longo["tabela"] = tabela

    nomes = longo.drop_duplicates(["nivel", "territorio_codigo"])[
        ["nivel", "territorio_codigo", "territorio_nome"]
    ]
    unidades = longo[["periodo", "produto", "variavel", "unidade"]].drop_duplicates()
    largo = longo.set_index(_INDICE + ["variavel"])["valor"].unstack("variavel").reset_index()
    largo.columns.name = None
    largo = largo.merge(nomes, on=["nivel", "territorio_codigo"], how="left", validate="m:1")
    return largo, unidades


def baixar_pam(
    config: dict[str, Any], *, refresh: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PAM de todas as tabelas do settings (temporárias, permanentes e milho por safra).

    Devolve (dados, unidades). `dados` tem uma linha por ano × território × produto, com
    quantidade (t), áreas (ha) e rendimento (kg/ha) = quantidade / área colhida. `unidades`
    lista a unidade publicada de cada variável, para checar mudanças de unidade.
    """
    cfg = config["producao"]["pam"]
    partes, unidades = [], []
    for t in cfg["tabelas"]:
        dados, unid = _baixar_tabela(
            t["tabela"],
            classificacao=t["classificacao"],
            variaveis=t["variaveis"],
            produtos=t["produtos"],
            niveis=cfg["niveis_territoriais"],
            inicio=cfg["inicio"],
            config=config,
            refresh=refresh,
        )
        partes.append(dados)
        unidades.append(unid.assign(tabela=t["tabela"]))
    pam = pd.concat(partes, ignore_index=True)
    pam.insert(0, "ano", pam.pop("periodo").astype(int))
    area = pam["area_colhida_ha"].where(pam["area_colhida_ha"] > 0)
    pam["rendimento_kg_ha"] = 1000 * pam["quantidade_t"] / area
    pam["rendimento_kg_ha"] = pam["rendimento_kg_ha"].replace([np.inf, -np.inf], np.nan)

    quebra = cfg.get("quantidade_fora_de_toneladas")
    if quebra:
        fora = pam["produto"].isin(quebra["produtos"]) & (pam["ano"] <= quebra["ate_ano"])
        pam.loc[fora, ["quantidade_t", "rendimento_kg_ha"]] = np.nan
        log.warning(
            "PAM: quantidade e rendimento de %s até %d marcados como ausentes (%d linhas): "
            "unidade publicada não era tonelada",
            ", ".join(quebra["produtos"]), quebra["ate_ano"], int(fora.sum()),
        )  # fmt: skip
    return pam, pd.concat(unidades, ignore_index=True)


def baixar_lspa(config: dict[str, Any], *, refresh: bool = False) -> pd.DataFrame:
    """LSPA: em cada mês (`mes`), a estimativa da safra do ano corrente, por UF e Brasil."""
    cfg = config["producao"]["lspa"]
    lspa, _ = _baixar_tabela(
        cfg["tabela"],
        classificacao=cfg["classificacao"],
        variaveis=cfg["variaveis"],
        produtos=cfg["produtos"],
        niveis=cfg["niveis_territoriais"],
        inicio=None,
        config=config,
        refresh=refresh,
    )
    lspa.insert(0, "mes", pd.to_datetime(lspa.pop("periodo"), format="%Y%m"))
    return lspa


def _valor_quantidade(config: dict[str, Any], refresh: bool) -> pd.DataFrame:
    """Valor da produção e quantidade (PAM e PPM) do Brasil e de cada UF, em formato largo."""
    cfg = config["precos_atacado"]["representatividade"]["anual"]
    partes = []
    for t in cfg["ibge"]["tabelas"]:
        dados, unidades = _baixar_tabela(
            t["tabela"],
            classificacao=t["classificacao"],
            variaveis=t["variaveis"],
            produtos=t["produtos"],
            niveis=cfg["ibge"]["niveis"],
            inicio=cfg["anos"][0],
            config=config,
            refresh=refresh,
        )
        valor = unidades[unidades["variavel"] == "valor_mil_reais"]["unidade"].unique()
        if set(valor) != {"Mil Reais"}:
            raise ValueError(f"tabela {t['tabela']}: valor da produção fora de mil reais: {valor}")
        unid = unidades[unidades["variavel"] == "quantidade"].groupby("produto")["unidade"]
        if (unid.nunique() > 1).any():
            raise ValueError(f"tabela {t['tabela']}: unidade da quantidade muda no período")
        dados["unidade_quantidade"] = dados["produto"].map(unid.first())
        partes.append(dados)
    return pd.concat(partes, ignore_index=True)


def preco_implicito(config: dict[str, Any], *, refresh: bool = False) -> pd.DataFrame:
    """Preço implícito ao produtor (valor da produção / quantidade), por ano e produto.

    Tabelas de `precos_atacado.representatividade.anual.ibge` (PAM e PPM). Devolve uma linha
    por ano × produto × território, com o Brasil, o PR e o "Brasil sem PR", cujo valor e
    quantidade são os do Brasil menos os do PR. `preco` está em mil R$ por unidade de
    quantidade (`unidade_quantidade`); só as razões e as variações são usadas.
    """
    ibge = _valor_quantidade(config, refresh)
    ibge = ibge[(ibge["nivel"] == "n1") | (ibge["territorio_codigo"] == "41")]  # Brasil e PR
    ibge["territorio"] = np.where(ibge["nivel"] == "n1", "brasil", "pr")
    ibge = ibge.assign(ano=ibge.pop("periodo").astype(int))
    largo = ibge.pivot_table(
        index=["ano", "produto", "unidade_quantidade"], columns="territorio",
        values=["valor_mil_reais", "quantidade"],
    )  # fmt: skip
    resto = (largo.xs("brasil", axis=1, level=1) - largo.xs("pr", axis=1, level=1)).assign(
        territorio="brasil_sem_pr"
    )
    saida = pd.concat(
        [ibge.set_index(["ano", "produto", "unidade_quantidade"])[
            ["territorio", "valor_mil_reais", "quantidade"]], resto]
    ).reset_index()  # fmt: skip
    saida["preco"] = saida["valor_mil_reais"] / saida["quantidade"]
    return saida.sort_values(["produto", "territorio", "ano"], ignore_index=True)


def preco_implicito_uf(config: dict[str, Any], *, refresh: bool = False) -> pd.DataFrame:
    """Como `preco_implicito`, por UF: ano, produto, uf (código IBGE), quantidade e preço.

    UFs sem produção num ano ficam com preço ausente (valor / 0 não é calculado)."""
    ibge = _valor_quantidade(config, refresh)
    ufs = ibge[ibge["nivel"] == "n3"].rename(columns={"territorio_codigo": "uf"})
    ufs = ufs.assign(ano=ufs.pop("periodo").astype(int))
    quantidade = ufs["quantidade"].where(ufs["quantidade"] > 0)
    ufs["preco"] = ufs["valor_mil_reais"] / quantidade
    return ufs[["ano", "produto", "uf", "territorio_nome", "quantidade", "preco"]]
