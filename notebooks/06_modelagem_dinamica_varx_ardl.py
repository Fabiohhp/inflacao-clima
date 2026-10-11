# %% [markdown]
# # 06 — Modelos dinâmicos: VECM e ARDL
#
# **Parte 7a (validação da fonte de preços no atacado):** não há série nacional mensal aberta,
# por produto, que cubra a janela. A candidata é a Seab-PR/Deral (via IPEADATA), mensal e
# completa, mas só do Paraná. Antes de usá-la, este teste mede quanto o Paraná se afasta do
# resto do Brasil, descontada a inflação geral (IPCA):
#
# - **anual (nacional):** preço implícito ao produtor do IBGE (valor da produção / quantidade;
#   PAM para milho e soja, PPM para leite e ovos), para o PR e para o "Brasil sem PR";
# - **mensal:** milho no atacado do PR × SP (Conab) e, de forma indireta para as carnes, o
#   IPCA de Curitiba × Brasil;
# - **retrato recente (só descritivo):** preços da Conab por UF, de 2025-10 a 2026-09.
#
# Comparações e critérios registrados no settings antes do cálculo
# (`precos_atacado.representatividade`). A fonte só é usada se passar no teste.
#
# **Parte 7b (VECM e ARDL):** com a fonte escolhida (milho no atacado da Seab-PR, R32), a
# relação entre o preço real do milho e os preços reais de aves e ovos e de carnes no IPCA:
# teste de Johansen (posto e caso determinístico pelo princípio de Pantula), VECM com posto 1 e
# ARDL com bounds test; elasticidade de longo prazo, velocidade de ajuste e multiplicadores de
# curto prazo com intervalos de 90%. Especificação e robustez registradas antes de estimar
# (`modelos.dinamicos`). Ressalvas centrais: R32, R33, R21.
#
# Rodar como script: `python notebooks/06_modelagem_dinamica_varx_ardl.py`.

# %%
import numpy as np
import pandas as pd
from statsmodels.tsa.coint_tables import c_sjt

from clima_inflacao import lakehouse, visual
from clima_inflacao.config import caminho_projeto, carregar_config, garantir_diretorios
from clima_inflacao.ingestao import conab, ipeadata, producao
from clima_inflacao.log import configurar_log
from clima_inflacao.modelos import dinamicos as dn
from clima_inflacao.modelos import eda
from clima_inflacao.modelos import representatividade as rp

config = carregar_config()
garantir_diretorios(config)
configurar_log(caminho_projeto("logs", config) / "06_dinamicos.log")
TABELAS = caminho_projeto("tabelas", config)
FIGURAS = caminho_projeto("figuras", config)
cfg = config["precos_atacado"]
rep = cfg["representatividade"]
CRITERIOS = rp.Criterios(**rep["criterios"])
pd.set_option("display.width", 170, "display.max_columns", 30)

# %% [markdown]
# ## Fontes

# %%
seab = ipeadata.series_mensais(list(cfg["series"]), config=config)
ibge = producao.preco_implicito(config)
conab_uf = conab.precos_mensais_uf(config)
ipca = lakehouse.ler("ipca", config)


def indice_ipca(area: str, categoria: str) -> pd.Series:
    serie = ipca[(ipca["area_id"] == area) & (ipca["categoria"] == categoria)]
    return serie.set_index("mes")["indice"].sort_index()


geral = indice_ipca(rep["deflator"]["area"], rep["deflator"]["categoria"])
INICIO = pd.Timestamp(rep["mensal"]["janela"]["inicio"])
FIM = pd.Timestamp(rep["mensal"]["janela"]["fim"])
ANOS = range(rep["anual"]["anos"][0], rep["anual"]["anos"][1] + 1)

janela = pd.date_range(INICIO, FIM, freq="MS")
fontes = []
for nome, s in cfg["series"].items():
    meta = ipeadata.metadados(s["codigo"], config=config)
    valida = seab[nome].dropna()
    fontes.append({
        "serie": nome, "codigo": s["codigo"], "fonte": meta["FNTSIGLA"],
        "nome_ipeadata": meta["SERNOME"], "unidade": s["unidade"],
        "inicio": valida.index.min().strftime("%Y-%m"), "fim": valida.index.max().strftime("%Y-%m"),
        "faltantes_na_janela": int(seab[nome].reindex(janela).isna().sum()),
    })  # fmt: skip
fontes = pd.DataFrame(fontes)
fontes.to_csv(TABELAS / "atacado_fontes.csv", index=False)
print(fontes.drop(columns="nome_ipeadata").to_string(index=False))

longo = seab.rename_axis("mes").reset_index().melt("mes", var_name="serie", value_name="preco")
lakehouse.salvar(longo.dropna(), "precos_atacado_candidatos", config)
lakehouse.salvar(ibge, "preco_implicito_ibge", config)

# %% [markdown]
# ## Anual: Seab-PR e IBGE (PR e Brasil sem PR)
#
# A Seab entra como média simples dos 12 meses do ano (só anos completos); o IBGE valoriza a
# safra a preços da colheita. Três comparações por produto: geografia (IBGE PR × IBGE Brasil
# sem PR), fonte (Seab PR × IBGE PR) e a decisiva (Seab PR × IBGE Brasil sem PR).

# %%
meses_por_ano = seab.groupby(seab.index.year).count()
seab_anual = seab.groupby(seab.index.year).mean().where(meses_por_ano == 12)
por_ano = geral.groupby(geral.index.year)
geral_anual = por_ano.mean().where(por_ano.count() == 12)
usadas_anual = list(rep["anual"]["seab"].values()) + list(rep["anual"]["seab_atacado"].values())
seab_anual = seab_anual.loc[list(ANOS), usadas_anual]
geral_anual = geral_anual.loc[list(ANOS)]
if seab_anual.isna().any().any() or geral_anual.isna().any():
    raise ValueError("ano incompleto na Seab ou no IPCA dentro da janela anual")

ibge_largo = ibge[ibge["ano"].isin(ANOS)].pivot(index="ano", columns=["produto", "territorio"],
                                                 values="preco")  # fmt: skip

comparacoes_anuais = []
for produto, nome_seab in rep["anual"]["seab"].items():
    comparacoes_anuais += [
        (produto, "geografia: IBGE PR × IBGE Brasil sem PR", ibge_largo[(produto, "pr")],
         ibge_largo[(produto, "brasil_sem_pr")]),
        (produto, "fonte: Seab PR × IBGE PR", seab_anual[nome_seab], ibge_largo[(produto, "pr")]),
        (produto, "decisiva: Seab PR × IBGE Brasil sem PR", seab_anual[nome_seab],
         ibge_largo[(produto, "brasil_sem_pr")]),
    ]  # fmt: skip
for produto, nome_seab in rep["anual"]["seab_atacado"].items():
    comparacoes_anuais.append(
        (f"{produto} (atacado)", "decisiva: Seab PR atacado × IBGE Brasil sem PR",
         seab_anual[nome_seab], ibge_largo[(produto, "brasil_sem_pr")])
    )  # fmt: skip

linhas = []
for produto, comparacao, pr, referencia in comparacoes_anuais:
    metricas = rp.comparar(pr, referencia, geral_anual, defasagem=1, hac_lags=2)
    linhas.append({"bloco": "anual", "produto": produto, "comparacao": comparacao, **metricas,
                   **rp.avaliar(metricas, CRITERIOS)})  # fmt: skip

# %% [markdown]
# ## Mensal: milho no atacado PR × SP e IPCA de Curitiba × Brasil (variações em 12 meses)

# %%
# Níveis dentro da janela (como no anual): a 1ª variação em 12 meses termina em 2001-09.
k = rep["mensal"]["defasagem_variacao_meses"]
geral_janela = geral.loc[INICIO:FIM]
for par in rep["mensal"]["pares"]:
    pr = seab[par["pr"]].loc[INICIO:FIM].dropna()
    referencia = seab[par["referencia"]].loc[INICIO:FIM].dropna()
    metricas = rp.comparar(pr, referencia, geral_janela, defasagem=k, hac_lags=k)
    linhas.append({"bloco": "mensal", "produto": "milho (atacado)",
                   "comparacao": "Seab PR × Conab SP", **metricas,
                   **rp.avaliar(metricas, CRITERIOS)})  # fmt: skip
for grupo in rep["mensal"]["ipca_curitiba"]:
    pr = indice_ipca("curitiba", grupo).loc[INICIO:FIM]
    referencia = indice_ipca("brasil", grupo).loc[INICIO:FIM]
    metricas = rp.comparar(pr, referencia, geral_janela, defasagem=k, hac_lags=k)
    linhas.append({"bloco": "mensal (varejo, indireto)", "produto": grupo,
                   "comparacao": "IPCA Curitiba × IPCA Brasil", **metricas,
                   **rp.avaliar(metricas, CRITERIOS)})  # fmt: skip

# %% [markdown]
# ## Diagnóstico pós-teste: resto do Brasil a pesos fixos (não muda o veredito)
#
# Registrado depois de ver o resultado (`composicao_fixa`): separa a mudança de composição da
# produção entre UFs de uma diferença de preço.

# %%
precos_uf = producao.preco_implicito_uf(config)
restos = {}
BASE = rep["anual"]["composicao_fixa"]["periodo_base"]
seab_diagnostico = {
    **rep["anual"]["seab"],
    "milho (atacado)": rep["anual"]["seab_atacado"]["milho"],
}
for produto, nome_seab in seab_diagnostico.items():
    cultura = produto.split(" ")[0]
    outras = precos_uf[(precos_uf["produto"] == cultura) & (precos_uf["uf"] != "41")]
    pesos = outras[outras["ano"].between(*BASE)].groupby("uf")["quantidade"].mean()
    largo = outras[outras["ano"].isin(ANOS)].pivot(index="ano", columns="uf", values="preco")
    resto = rp.indice_pesos_fixos(largo, pesos)
    restos[cultura] = resto
    pares = [("decisiva (pesos fixos): Seab PR × resto do Brasil", seab_anual[nome_seab])]
    if produto == cultura:
        pares.insert(0, ("geografia (pesos fixos): IBGE PR × resto do Brasil",
                         ibge_largo[(cultura, "pr")]))  # fmt: skip
    for comparacao, pr in pares:
        metricas = rp.comparar(pr, resto, geral_anual, defasagem=1, hac_lags=2)
        linhas.append({"bloco": "anual (pós-teste, pesos fixos)", "produto": produto,
                       "comparacao": comparacao, **metricas, **rp.avaliar(metricas, CRITERIOS),
                       "ufs_pesos_fixos": len(resto.attrs["ufs"]),
                       "peso_fora_pesos_fixos": resto.attrs["peso_fora"]})  # fmt: skip

resultado = pd.DataFrame(linhas)
resultado["inicio"] = resultado["inicio"].astype(str).str[:7]
resultado["fim"] = resultado["fim"].astype(str).str[:7]
resultado["divergencia_fracao_inflacao"] = (
    resultado["divergencia_acumulada"] / resultado["inflacao_acumulada"]
)
resultado["diferenca_fracao_inflacao"] = resultado["diferenca_media"] / resultado["inflacao_media"]
resultado.to_csv(TABELAS / "atacado_representatividade.csv", index=False)
colunas = ["bloco", "produto", "comparacao", "n_variacoes", "correlacao_real",
           "correlacao_nominal", "inclinacao", "diferenca_media", "inflacao_media",
           "divergencia_acumulada", "inflacao_acumulada", "tendencia_razao_ano", "tendencia_p",
           "adf_p", "chega_perto"]  # fmt: skip
print(resultado[colunas].round(3).to_string(index=False))

# %% [markdown]
# ## Retrato recente: Conab por UF (2025-10 a 2026-09; só descritivo)
#
# UFs com os 12 meses; linhas repetidas por UF e mês (ex.: milho no atacado em MT) entram pela
# média geométrica, e a tabela conta quantas foram juntadas. A concordância Seab × Conab no PR
# usa os meses comuns às duas fontes.

# %%
retratos = []
for p in rep["conab_uf"]["produtos"]:
    filtro = (
        (conab_uf["produto"] == p["produto"]) & (conab_uf["nivel"] == p["nivel"])
        & (conab_uf["classificacao"] == p["classificacao"]) & (conab_uf["valor"] > 0)
    )  # fmt: skip
    dados = conab_uf[filtro]
    precos = np.exp(np.log(dados["valor"]).groupby([dados["mes"], dados["uf"]]).mean()).unstack()
    retrato = rp.retrato_uf(precos)
    seab_pr = seab[p["seab"]]
    concord = rp.concordancia(seab_pr, precos["PR"])
    retratos.append({
        "produto": p["nome"], **retrato,
        "linhas_repetidas_juntadas": int(dados["repetida"].sum()),
        "seab_conab_meses": concord["n_meses"],
        "seab_conab_correlacao_variacoes": concord["correlacao_variacoes"],
        "seab_conab_razao_cv_pct": concord["razao_cv_pct"],
    })  # fmt: skip
retratos = pd.DataFrame(retratos)
retratos.to_csv(TABELAS / "atacado_conab_uf.csv", index=False)
print(retratos.round(3).to_string(index=False))

# %% [markdown]
# ## Figura: preços reais (início = 100)

# %%
SERIES_FIGURA = {
    "seab_pr": "Paraná (Seab)",
    "ibge_pr": "Paraná (IBGE, preço implícito)",
    "referencia": "Brasil sem PR (IBGE) / São Paulo (Conab, mensal)",
    "pesos_fixos": "resto do Brasil a pesos fixos (pós-teste)",
}
partes = []
for painel, nome_seab in {
    **rep["anual"]["seab"],
    "milho_atacado_anual": "milho_atacado_pr",
}.items():
    produto = painel.split("_")[0]
    reais = {
        "seab_pr": seab_anual[nome_seab] / geral_anual,
        "ibge_pr": ibge_largo[(produto, "pr")] / geral_anual,
        "referencia": ibge_largo[(produto, "brasil_sem_pr")] / geral_anual,
        "pesos_fixos": restos[produto] / geral_anual,
    }
    if painel == "milho_atacado_anual":
        del reais["ibge_pr"]  # o IBGE só tem o preço ao produtor; está no painel do milho
    for serie, valores in reais.items():
        partes.append(pd.DataFrame({"painel": painel, "serie": serie, "periodo": valores.index,
                                    "valor": 100 * valores / valores.iloc[0]}))  # fmt: skip
for serie, nome in [("seab_pr", "milho_atacado_pr"), ("referencia", "milho_atacado_sp")]:
    real = (seab[nome] / geral).loc[INICIO:FIM]
    partes.append(pd.DataFrame({"painel": "milho_atacado_mensal", "serie": serie,
                                "periodo": real.index,
                                "valor": 100 * real / real.iloc[0]}))  # fmt: skip
figura_tabela = pd.concat(partes, ignore_index=True)
figura_tabela.to_csv(TABELAS / "atacado_precos_reais.csv", index=False)

PAINEIS = ["milho_atacado_anual", "milho", "milho_atacado_mensal", "soja", "leite", "ovos"]
decisiva = resultado[(resultado["bloco"] == "anual")
                     & resultado["comparacao"].str.startswith("decisiva: Seab PR")]  # fmt: skip
decisiva = decisiva.set_index("produto")
mensal_milho = resultado[resultado["comparacao"] == "Seab PR × Conab SP"].iloc[0]


def titulo_painel(produto: str) -> str:
    linha = decisiva.loc[produto]
    return (f"{produto.capitalize()} ao produtor (anual): correlação real "
            f"{visual.numero(linha['correlacao_real'])}")  # fmt: skip


TITULOS = {p: titulo_painel(p) for p in rep["anual"]["seab"]}
TITULOS["milho_atacado_anual"] = (
    "Milho no atacado (anual): correlação real "
    f"{visual.numero(decisiva.loc['milho (atacado)', 'correlacao_real'])}"
)
TITULOS["milho_atacado_mensal"] = (
    "Milho no atacado, PR × SP (mensal): correlação real "
    f"{visual.numero(mensal_milho['correlacao_real'])}"
)
fig = visual.precos_paineis(
    figura_tabela, PAINEIS, TITULOS, SERIES_FIGURA,
    titulo="Preço real no Paraná e no resto do Brasil",
    nota=("Preços descontados do IPCA geral, índice no primeiro período = 100. Anual: Seab-PR "
          "(média dos 12 meses) e preço implícito do IBGE (valor da produção / quantidade; PAM "
          "para milho e soja, PPM para leite e ovos). Mensal: milho no atacado, Seab-PR e Conab "
          "SP (faltam 48 meses a partir de 2016-03). Pesos fixos: média das UFs sem o PR com o "
          "peso de 2003-2007 (diagnóstico pós-teste). Correlação das variações reais (anuais ou "
          "em 12 meses): Seab PR × IBGE Brasil sem PR no anual; PR × SP no mensal. Tabelas: "
          "atacado_precos_reais.csv e atacado_representatividade.csv."),
    rotulo_y="preço real (início = 100)",
)  # fmt: skip
fig.savefig(FIGURAS / "atacado_pr_brasil.png", dpi=150)


# %% [markdown]
# # Parte 7b — VECM e ARDL: milho no atacado (PR) → aves e carnes no IPCA
#
# Preços em log, descontados do IPCA geral (nominais na robustez). Amostra efetiva de 2000-09
# a 2023-06; as defasagens vêm dos meses anteriores. Valores críticos do caso 2 simulados com
# a semente do projeto.

# %%
dcfg = config["modelos"]["dinamicos"]
SEMENTE = config["projeto"]["semente_aleatoria"]
NIVEL = dcfg["nivel_confianca"]
base = lakehouse.ler("base_analise_brasil", config).set_index("mes")
nominais = pd.DataFrame({
    "log_aves_ovos": base["log_aves_ovos"], "log_carnes": base["log_carnes"],
    "milho_atacado_pr": np.log(seab["milho_atacado_pr"]),
    "milho_atacado_sp": np.log(seab["milho_atacado_sp"]),
    "log_icbr_agro": base["log_icbr_agro"], "log_oleo_diesel": base["log_oleo_diesel"],
})  # fmt: skip
reais = nominais.sub(base[dcfg["deflator"]], axis=0)
INI7 = pd.Timestamp(dcfg["janela"]["inicio"])
FIM7 = pd.Timestamp(dcfg["janela"]["fim"])
MAX_K = dcfg["johansen"]["max_k_ar_diff"]
MAX_ARDL = dcfg["ardl"]["max_lags"]

sim = dcfg["johansen"]["simulacao_valores_criticos"]
cv_caso1 = dn.simular_valores_criticos(1, [1, 2, 3], t=sim["t"], replicas=sim["replicas"],
                                       semente=SEMENTE)  # fmt: skip
cv_caso2 = dn.simular_valores_criticos(2, [1, 2, 3], t=sim["t"], replicas=sim["replicas"],
                                       semente=SEMENTE)  # fmt: skip
validacao = cv_caso1.assign(**{
    f"tabela_{c}": [c_sjt(d, -1)[i] for d in cv_caso1.index]
    for i, c in enumerate(["cv_90", "cv_95", "cv_99"])
})  # fmt: skip
erro_max = (validacao["cv_95"] / validacao["tabela_cv_95"] - 1).abs().max()
print(f"Simulador (caso 1) × tabela: erro máximo no 95% = {100 * erro_max:.1f}%")
if erro_max > 0.03:
    raise ValueError("simulador de valores críticos não reproduz a tabela do caso 1")
pd.concat({"caso_1_validacao": validacao, "caso_2": cv_caso2}, names=["tabela"]).to_csv(
    TABELAS / "vecm_ardl_valores_criticos.csv"
)

# %% [markdown]
# ## Especificações registradas

# %%
GRAOS = {"milho_atacado_pr": "milho_atacado_pr", "milho_atacado_sp": "milho_atacado_sp",
         "log_icbr_agro": "log_icbr_agro"}  # fmt: skip
especs = []
for cadeia, c in dcfg["cadeias"].items():
    especs.append({"cadeia": cadeia, "espec": "principal", "varejo": c["varejo"],
                   "grao": c["grao"], "insumos": [], "real": True, "inicio": INI7, "fim": FIM7,
                   "ardl_caso": None, "vecm_det": None})  # fmt: skip
    for r in dcfg["robustez"]:
        janela = r.get("janela", {})
        especs.append({
            "cadeia": cadeia, "espec": r["nome"], "varejo": c["varejo"],
            "grao": r.get("grao", c["grao"]), "insumos": r.get("insumos", []),
            "real": r.get("deflator", "sim") is not None,
            "inicio": pd.Timestamp(janela.get("inicio", INI7)),
            "fim": pd.Timestamp(janela.get("fim", FIM7)),
            "ardl_caso": r.get("ardl_caso"), "vecm_det": r.get("vecm_deterministico"),
        })  # fmt: skip


def dados_espec(e: dict) -> pd.DataFrame:
    """Níveis (log) da especificação, com 13 meses antes do início para as defasagens."""
    fonte = reais if e["real"] else nominais
    colunas = {e["varejo"]: "varejo", e["grao"]: "grao"} | {i: i.removeprefix("log_")
                                                           for i in e["insumos"]}  # fmt: skip
    dados = fonte[list(colunas)].rename(columns=colunas)
    dados = dados.loc[e["inicio"] - pd.DateOffset(months=MAX_K + 1) : e["fim"]]
    if dados.isna().any().any():
        raise ValueError(f"faltantes em {e['cadeia']}/{e['espec']}")
    return dados.asfreq("MS")


# %% [markdown]
# ## Ordem de integração (regra da Parte 4, séries dessazonalizadas por STL, na janela de cada
# especificação)

# %%
integracao = []
vistos = set()
for e in especs:
    dados = dados_espec(e).loc[e["inicio"] :]
    for coluna in dados.columns:
        nome = {"varejo": e["varejo"], "grao": e["grao"]}.get(coluna, coluna)
        chave = (nome, e["real"], e["inicio"], e["fim"])
        if chave in vistos:
            continue
        vistos.add(chave)
        ajustada = eda.dessazonalizar_stl(dados[coluna])["dessazonalizado"]
        ordem, detalhe = eda.ordem_integracao(ajustada, dcfg["ordem_integracao"]["tendencia_nivel"],
                                              dcfg["ordem_integracao"]["alfa"])  # fmt: skip
        nivel = detalhe.iloc[0]
        integracao.append({"serie": nome, "preco": "real" if e["real"] else "nominal",
                           "inicio": f"{e['inicio']:%Y-%m}", "fim": f"{e['fim']:%Y-%m}",
                           "ordem": ordem, "adf_p_nivel": nivel["adf_p"],
                           "kpss_p_nivel": nivel["kpss_p"]})  # fmt: skip
integracao = pd.DataFrame(integracao)
integracao.to_csv(TABELAS / "vecm_ardl_integracao.csv", index=False)
print(integracao.to_string(index=False))
if integracao["ordem"].str.contains(">").any() or (integracao["ordem"] == "I(2)").any():
    print("ATENÇÃO: série I(2) ou acima — o bounds test não vale para ela")

# %% [markdown]
# ## Johansen (Pantula), VECM e ARDL

# %%
resumos, testes_johansen, multiplicadores = [], [], []
postos_principais = {}
for e in especs:
    dados = dados_espec(e)
    linha = {"cadeia": e["cadeia"], "espec": e["espec"]}
    if e["vecm_det"] is None:
        k = dn.selecionar_k_ar_diff(dados, MAX_K, dcfg["johansen"]["criterio_lags"])
        amostra = dados.loc[e["inicio"] - pd.DateOffset(months=k + 1) :]
        r, caso, testes = dn.pantula(amostra, k, cv_caso2)
        testes_johansen.append(testes.assign(cadeia=e["cadeia"], espec=e["espec"], k_ar_diff=k))
        if e["espec"] == "principal":
            postos_principais[e["cadeia"]] = (r, caso, k)
        det = dcfg["vecm"]["deterministico_por_caso"].get(caso) if caso else None
    else:  # tendência na relação: posto e defasagens do Johansen principal
        r, caso, k = postos_principais[e["cadeia"]]
        amostra = dados.loc[e["inicio"] - pd.DateOffset(months=k + 1) :]
        det = e["vecm_det"]
    linha.update({"k_ar_diff": k, "posto": r, "caso": caso, "vecm_deterministico": det})
    if r == 1:
        linha.update({f"vecm_{c}": v for c, v in dn.vecm(amostra, k, det, NIVEL).items()})

    x = dados.drop(columns="varejo").loc[e["inicio"] - pd.DateOffset(months=MAX_ARDL) :]
    y = dados["varejo"].loc[x.index[0] :]
    caso_bounds = e["ardl_caso"] or (caso if caso in (2, 3) else 3)
    tendencia = "ct" if caso_bounds in (4, 5) else "c"
    resumo_ardl, modelo = dn.ardl(
        y, x, max_lags=MAX_ARDL, criterio=dcfg["ardl"]["criterio_lags"], tendencia=tendencia,
        caso_bounds=caso_bounds, hac_lags=dcfg["ardl"]["erros"]["lags"], nivel=NIVEL,
        ordem_minima=dcfg["ardl"]["ordem_minima"],
    )  # fmt: skip
    linha.update(
        {"ardl_caso_bounds": caso_bounds, **{f"ardl_{c}": v for c, v in resumo_ardl.items()}}
    )
    linha["ardl_leitura"] = dn.leitura_bounds(
        resumo_ardl["bounds_f"], resumo_ardl["bounds_cv5_i0"], resumo_ardl["bounds_cv5_i1"]
    )
    mult = dn.multiplicadores_simulados(
        modelo, "grao", horizonte=dcfg["multiplicadores"]["horizonte_max"],
        simulacoes=dcfg["multiplicadores"]["simulacoes"],
        hac_lags=dcfg["ardl"]["erros"]["lags"], nivel=NIVEL, semente=SEMENTE,
    )  # fmt: skip
    multiplicadores.append(mult.assign(cadeia=e["cadeia"], espec=e["espec"]))
    longo = mult[mult["horizonte"] == "longo"].iloc[0]
    curto = mult[mult["horizonte"] != "longo"].set_index("horizonte")
    # Meses até atingir 50% e 90% do longo prazo: só quando o longo prazo é positivo e o seu
    # intervalo exclui zero (senão a fração não tem sentido).
    linha["ardl_longo_sim"], linha["ardl_longo_sim_li"], linha["ardl_longo_sim_ls"] = longo[
        ["coef", "li", "ls"]]  # fmt: skip
    for frac in (0.5, 0.9):
        atingiu = curto.index[curto["coef"] >= frac * longo["coef"]] if longo["li"] > 0 else []
        linha[f"ardl_h_{int(100 * frac)}pct_longo"] = int(atingiu[0]) if len(atingiu) else np.nan
    linha["ardl_impacto"], linha["ardl_impacto_li"], linha["ardl_impacto_ls"] = curto.loc[
        0, ["coef", "li", "ls"]]  # fmt: skip
    linha["ardl_h12"], linha["ardl_h12_li"], linha["ardl_h12_ls"] = curto.loc[
        12, ["coef", "li", "ls"]]  # fmt: skip
    resumos.append(linha)

resumo = pd.DataFrame(resumos)
resumo.to_csv(TABELAS / "vecm_ardl_resumo.csv", index=False)
lakehouse.salvar(resumo, "vecm_ardl_resumo", config)
pd.concat(testes_johansen, ignore_index=True).to_csv(TABELAS / "vecm_ardl_johansen.csv",
                                                     index=False)  # fmt: skip
multiplicadores = pd.concat(multiplicadores, ignore_index=True)
multiplicadores["horizonte"] = multiplicadores["horizonte"].astype(str)
multiplicadores.to_csv(TABELAS / "vecm_ardl_multiplicadores.csv", index=False)
lakehouse.salvar(multiplicadores, "vecm_ardl_multiplicadores", config)

colunas = ["cadeia", "espec", "k_ar_diff", "posto", "caso", "vecm_elasticidade_grao",
           "vecm_elasticidade_grao_li", "vecm_elasticidade_grao_ls", "vecm_alpha_varejo",
           "vecm_alpha_varejo_li", "vecm_alpha_varejo_ls", "vecm_alpha_grao_p",
           "vecm_meia_vida_meses", "vecm_brancura_p"]  # fmt: skip
print(resumo.reindex(columns=colunas).round(3).to_string(index=False))
colunas = ["cadeia", "espec", "ardl_p", "ardl_q_grao", "ardl_caso_bounds", "ardl_bounds_f",
           "ardl_bounds_cv5_i0", "ardl_bounds_cv5_i1", "ardl_leitura", "ardl_elasticidade_grao",
           "ardl_elasticidade_grao_li", "ardl_elasticidade_grao_ls", "ardl_velocidade",
           "ardl_velocidade_li", "ardl_velocidade_ls", "ardl_meia_vida_meses", "ardl_impacto",
           "ardl_h12", "ardl_h_50pct_longo", "ardl_h_90pct_longo"]  # fmt: skip
print(resumo.reindex(columns=colunas).round(3).to_string(index=False))


# %% [markdown]
# ## Figuras: multiplicadores (principal) e elasticidade de longo prazo (aves e ovos)

# %%
NOMES_CADEIAS = {"aves_ovos": "Aves e ovos", "carnes": "Carnes"}
principais = resumo[resumo["espec"] == "principal"].set_index("cadeia")
mult_principal = multiplicadores[(multiplicadores["espec"] == "principal")
                                 & (multiplicadores["horizonte"] != "longo")].copy()  # fmt: skip
mult_principal["painel"] = mult_principal["cadeia"]
mult_principal["horizonte"] = mult_principal["horizonte"].astype(int)
longos = []
for cadeia, linha in principais.iterrows():
    if linha["ardl_leitura"] == "relação de longo prazo":
        longos.append({"painel": cadeia, "coef": linha["ardl_longo_sim"],
                       "li": linha["ardl_longo_sim_li"],
                       "ls": linha["ardl_longo_sim_ls"]})  # fmt: skip
longos = pd.DataFrame(longos, columns=["painel", "coef", "li", "ls"])


def titulo_cadeia(cadeia: str) -> str:
    linha = principais.loc[cadeia]
    if linha["ardl_leitura"] == "relação de longo prazo":
        return (f"{NOMES_CADEIAS[cadeia]}: longo prazo {visual.numero(linha['ardl_longo_sim'])} "
                f"(meia-vida de {linha['ardl_meia_vida_meses']:.0f} meses)")  # fmt: skip
    return f"{NOMES_CADEIAS[cadeia]}: sem relação de longo prazo (bounds test)"


fig = visual.multiplicadores_paineis(
    mult_principal, longos, list(dcfg["cadeias"]),
    {c: titulo_cadeia(c) for c in dcfg["cadeias"]},
    titulo="Repasse do preço do milho no atacado aos preços de carnes no IPCA",
    nota=("Multiplicador dinâmico acumulado do ARDL (preços reais em log, dummies de mês): "
          "variação % do preço no varejo por 1% de alta permanente do milho no atacado (Seab-PR). "
          "Bandas de 90% por 5.000 sorteios dos parâmetros (erros HAC). Amostra 2000-09 a "
          "2023-06. Tracejado: multiplicador de longo prazo, só onde o bounds test indica relação "
          "de longo prazo. Tabela: vecm_ardl_multiplicadores.csv."),
    rotulo_y="variação acumulada (% por 1% no milho)",
)  # fmt: skip
fig.savefig(FIGURAS / "vecm_ardl_multiplicadores.png", dpi=150)

ROTULOS_ESPEC = {
    "principal": "principal (milho PR, real)", "milho_sp": "milho de SP (2000-09 a 2016-02)",
    "icbr_agro": "IC-Br Agropecuária no lugar do milho", "nominal": "preços nominais",
    "tendencia": "tendência na relação de longo prazo", "com_diesel": "+ diesel (insumo)",
}  # fmt: skip
floresta = []
for _, linha in resumo[resumo["cadeia"] == "aves_ovos"].iterrows():
    for metodo, prefixo in [("vecm", "vecm_elasticidade_grao"), ("ardl", "ardl_elasticidade_grao")]:
        floresta.append({"painel": "aves_ovos", "modelo": linha["espec"], "choque": metodo,
                         "coef": linha.get(prefixo), "li_90": linha.get(f"{prefixo}_li"),
                         "ls_90": linha.get(f"{prefixo}_ls")})  # fmt: skip
floresta = pd.DataFrame(floresta)
floresta.to_csv(TABELAS / "vecm_ardl_longo_prazo.csv", index=False)
fig = visual.floresta_grade(
    floresta, ["aves_ovos"], {"aves_ovos": "Aves e ovos"}, list(ROTULOS_ESPEC), ROTULOS_ESPEC,
    {"vecm": "VECM (Johansen)", "ardl": "ARDL"},
    titulo="Elasticidade de longo prazo do preço de aves e ovos ao preço do milho",
    nota=("Variação % de longo prazo do preço de aves e ovos no IPCA por 1% de alta permanente "
          "do milho; traço = intervalo de 90%. VECM só onde o Johansen indica posto 1 (o IC-Br "
          "deu posto 0). Na especificação de SP o milho real é I(0) na janela. Carnes: sem "
          "relação de longo prazo em todas as especificações. Tabela: vecm_ardl_longo_prazo.csv."),
    rotulo_x="elasticidade de longo prazo", ncols=1,
)  # fmt: skip
fig.savefig(FIGURAS / "vecm_ardl_longo_prazo.png", dpi=150)
