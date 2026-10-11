# %% [markdown]
# # 04 — EDA e decomposição temporal
#
# Monta a base de análise mensal do Brasil (preços do IPCA, controles, ENSO, CWI e cinturão
# verde), dessazonaliza os preços (STL; o X-13 não está instalado, ressalva R14), testa a
# ordem de integração (ADF e KPSS), calcula a correlação cruzada pré-branqueada clima x
# inflação e a coerência wavelet. Tudo na janela de estimação (1999-08 a 2023-06, ressalva R1).
#
# Pares clima x preço e janelas de defasagem da hipótese: definidos no settings antes do
# cálculo (`eda.pares`, `hipoteses_defasagens_meses`).
#
# Rodar como script: `python notebooks/04_eda_e_decomposicao_temporal.py`.

# %%
import numpy as np
import pandas as pd

from clima_inflacao import lakehouse, visual
from clima_inflacao.config import caminho_projeto, carregar_config, garantir_diretorios
from clima_inflacao.log import configurar_log
from clima_inflacao.modelos import eda

config = carregar_config()
garantir_diretorios(config)
configurar_log(caminho_projeto("logs", config) / "04_eda.log")
TABELAS = caminho_projeto("tabelas", config)
FIGURAS = caminho_projeto("figuras", config)
cfg = config["eda"]
hipoteses = config["hipoteses_defasagens_meses"]
amostra = config["periodos"]["amostra_estimacao"]
JANELA = (pd.Timestamp(amostra["inicio"]), pd.Timestamp(amostra["fim"]))
pd.set_option("display.width", 170, "display.max_columns", 30)

# %% [markdown]
# ## Base de análise mensal (Brasil)

# %%
ipca = lakehouse.ler("ipca", config)
brasil = ipca[ipca["area_id"] == "brasil"]
grupos = list(config["ipca"]["categorias"])
log_precos = np.log(brasil.pivot(index="mes", columns="categoria", values="indice")[grupos])
log_precos["oleo_diesel"] = np.log(
    lakehouse.ler("diesel_ipca", config).query("area_id == 'brasil'").set_index("mes")["indice"]
)
log_precos = log_precos.add_prefix("log_")

macro = lakehouse.ler("macro_mensal", config).pivot(index="mes", columns="serie", values="valor")
controles = pd.DataFrame(
    {"log_ptax": np.log(macro["ptax_venda"]), "log_icbr_agro": np.log(macro["icbr_agropecuaria"])}
)
enso = lakehouse.ler("enso", config).pivot(index="mes", columns="indice", values="valor")

cwi_mensal = lakehouse.ler("cwi_mensal", config).query("versao == 'defasado'")
colunas_cwi = {}
for (regiao, cultura), grupo in cwi_mensal.groupby(["regiao", "cultura"]):
    sufixo = "" if regiao == "brasil" else f"_{regiao}"
    for spi in config["espacial"]["cwi"]["indicadores"]:
        colunas_cwi[f"cwi_{cultura}_{spi}{sufixo}"] = grupo.set_index("mes")[spi]
cwi_largo = pd.DataFrame(colunas_cwi)

# Cinturão verde médio: as 11 áreas com IPCA desde antes de 1999 (as demais entram depois).
entrada = ipca.groupby("area_id")["mes"].min()
areas_longas = [a for a in entrada.index if entrada[a] <= JANELA[0] and a != "brasil"]
cinturao = lakehouse.ler("cinturao_verde_mensal", config)
cinturao = cinturao[
    (cinturao["raio_km"] == cfg["raio_cinturao_km"]) & cinturao["area_id"].isin(areas_longas)
]
cinturao_medio = cinturao.groupby("mes")[["spi_3", "spi_6"]].mean().add_prefix("cinturao_")
print(f"Cinturão verde médio ({cfg['raio_cinturao_km']} km): {len(areas_longas)} áreas")

base = pd.concat([log_precos, controles, enso, cwi_largo, cinturao_medio], axis=1, sort=True)
base = base[base.index >= pd.Timestamp("1997-01-01")]
base.index.name = "mes"

# %% [markdown]
# ## Dessazonalização (STL) dos preços, na janela

# %%
na_janela = (base.index >= JANELA[0]) & (base.index <= JANELA[1])
series_preco = [*grupos, "oleo_diesel"]
perfis, forca = [], {}
for grupo in series_preco:
    serie = 100 * base.loc[na_janela, f"log_{grupo}"]
    stl = eda.dessazonalizar_stl(serie, cfg["stl"]["periodo"], cfg["stl"]["robusto"])
    base.loc[na_janela, f"log_sa_{grupo}"] = stl["dessazonalizado"] / 100
    base.loc[na_janela, f"infl_sa_{grupo}"] = stl["dessazonalizado"].diff()
    base.loc[na_janela, f"infl_{grupo}"] = serie.diff()
    perfil = stl["sazonal"].groupby(stl.index.month).mean()
    perfis.append(
        pd.DataFrame(
            {"grupo": grupo, "mes_do_ano": perfil.index, "fator_sazonal": perfil.to_numpy()}
        )
    )
    variancia = (stl["sazonal"] + stl["residuo"]).var()
    forca[grupo] = max(0.0, 1 - stl["residuo"].var() / variancia)
perfil_sazonal = pd.concat(perfis, ignore_index=True)
perfil_sazonal.to_csv(TABELAS / "stl_perfil_sazonal.csv", index=False)
forca_sazonal = pd.Series(forca, name="forca_sazonal").round(3)
forca_sazonal.to_csv(TABELAS / "stl_forca_sazonal.csv")
print("Força da sazonalidade (0 a 1; Wang, Smith e Hyndman, 2006):")
print(forca_sazonal.sort_values(ascending=False).to_string())
amplitude = (
    perfil_sazonal.groupby("grupo")["fator_sazonal"].agg(lambda s: s.max() - s.min()).round(2)
)
print("Amplitude do fator sazonal (pontos de log x 100):", amplitude.to_dict())
fig = visual.perfil_sazonal(
    perfil_sazonal,
    "Fator sazonal médio por mês do ano (STL, pontos de log x 100; escala própria por painel)",
)
fig.savefig(FIGURAS / "stl_perfil_sazonal.png")
lakehouse.salvar(base.reset_index(), "base_analise_brasil", config)

# %% [markdown]
# ## Ordem de integração (ADF e KPSS)
#
# Níveis em log com constante e tendência; taxas, SPI/CWI e ENSO com constante. Regra: a
# série diferenciada d vezes é estacionária se o ADF rejeita e o KPSS não rejeita (5%).

# %%
especificacao = (
    [(f"log_sa_{g}", "ct") for g in series_preco]
    + [(f"infl_sa_{g}", "c") for g in series_preco]
    + [("log_ptax", "ct"), ("log_icbr_agro", "ct"), ("oni", "c"), ("mei_v2", "c")]
    + [(c, "c") for c in ["cwi_soja_spi_3", "cwi_soja_spi_6", "cwi_milho_spi_6",
                          "cwi_milho_2a_safra_spi_6", "cinturao_spi_3", "cinturao_spi_6"]]
)  # fmt: skip
linhas, detalhes = [], []
for coluna, tendencia in especificacao:
    ordem, detalhe = eda.ordem_integracao(base.loc[na_janela, coluna], tendencia, cfg["alfa"])
    nivel = detalhe.iloc[0]
    linhas.append(
        {
            "serie": coluna, "tendencia_nivel": tendencia, "ordem_integracao": ordem,
            "adf_p_nivel": round(nivel["adf_p"], 4), "kpss_p_nivel": round(nivel["kpss_p"], 3),
            "conclusao_nivel": nivel["conclusao"], "n": int(nivel["n"]),
        }
    )  # fmt: skip
    detalhes.append(detalhe.assign(serie=coluna))
integracao = pd.DataFrame(linhas)

# Refinamento sistemático: em toda série que a regra básica não classificou como I(0),
# Zivot-Andrews (quebra de nível) e ADF/KPSS com tendência. Em séries com constante (taxas,
# SPI/CWI, ENSO), isso separa raiz unitária de tendência ou quebra. Um log-nível I(2) cuja
# inflação é I(0) com quebra ou tendência é lido como I(1).
diagnosticos = {}
for linha in integracao.itertuples(index=False):
    if linha.ordem_integracao != "I(0)":
        diagnosticos[linha.serie] = eda.diagnostico_quebra_tendencia(
            base.loc[na_janela, linha.serie]
        )
leituras = {}
for linha in integracao.itertuples(index=False):
    diag = diagnosticos.get(linha.serie)
    leituras[linha.serie] = (
        eda.leitura_integracao(linha.ordem_integracao, linha.tendencia_nivel, diag, cfg["alfa"])
        if diag else linha.ordem_integracao
    )  # fmt: skip
for serie, leitura in list(leituras.items()):
    if serie.startswith("log_sa_") and leitura.startswith("I(2)"):
        inflacao = leituras.get(serie.replace("log_sa_", "infl_sa_"), "")
        if inflacao.startswith("I(0)"):
            leituras[serie] = f"I(1) (inflação {inflacao})"
integracao["leitura_final"] = integracao["serie"].map(leituras)
for coluna in ["za_p", "za_quebra", "adf_ct_p", "kpss_ct_p"]:
    integracao[coluna] = integracao["serie"].map(lambda s, c=coluna: diagnosticos.get(s, {}).get(c))
integracao.to_csv(TABELAS / "ordem_integracao.csv", index=False)
pd.concat(detalhes).to_csv(TABELAS / "ordem_integracao_detalhe.csv", index=False)
print(
    integracao[["serie", "ordem_integracao", "leitura_final", "za_p", "za_quebra"]].to_string(
        index=False
    )
)
print("Contagem (regra básica):", integracao["ordem_integracao"].value_counts().to_dict())
print("Contagem (leitura final):", integracao["leitura_final"].value_counts().to_dict())

# %% [markdown]
# ## Correlação cruzada pré-branqueada: clima em t−k × inflação dessazonalizada em t
#
# Sinal esperado negativo (seca, SPI < 0, seguida de alta de preço). Bandas completas:
# pontual de 95% e Bonferroni sobre as 19 defasagens; com 12 pares, ~11 barras passam da
# banda pontual por acaso.

# %%
tabelas_ccf, avaliacoes = [], []
for par in cfg["pares"]:
    grupo, clima = par["grupo"], par["clima"]
    x = base.loc[na_janela, clima]
    y = base.loc[na_janela, f"infl_sa_{grupo}"]
    ccf = eda.ccf_prebranqueada(x, y, cfg["defasagem_max_meses"], cfg["alfa"])
    tabelas_ccf.append(ccf.assign(grupo=grupo, clima=clima))
    janela_h = hipoteses.get(grupo)
    avaliacoes.append(
        {"grupo": grupo, "clima": clima, "janela_hipotese": str(janela_h) if janela_h else "—",
         **eda.avaliar_defasagens(ccf, janela_h)}
    )  # fmt: skip
ccf_todas = pd.concat(tabelas_ccf, ignore_index=True)
ccf_todas.to_csv(TABELAS / "ccf_prebranqueada.csv", index=False)
lags = pd.DataFrame(avaliacoes)
lags.to_csv(TABELAS / "lags_candidatos.csv", index=False)
print(lags.to_string(index=False))
acima = (ccf_todas["r"].abs() > ccf_todas["banda_pontual"]).sum()
print(f"Barras acima da banda pontual: {acima} de {len(ccf_todas)} (esperado por acaso: "
      f"{0.05 * len(ccf_todas):.0f})")  # fmt: skip
fig = visual.ccf_paineis(
    ccf_todas, hipoteses, "Correlação cruzada pré-branqueada: clima em t−k × inflação em t"
)
fig.savefig(FIGURAS / "ccf_prebranqueada.png")

# %% [markdown]
# ## Coerência wavelet
#
# Onde, no tempo e na frequência, clima e inflação co-variam. Significância a 95% por Monte
# Carlo contra ruído vermelho (AR1), com a semente do projeto.

# %%
semente = config["projeto"]["semente_aleatoria"]
tabelas_wct = []
for par in cfg["wavelet"]["pares"]:
    grupo, clima = par["grupo"], par["clima"]
    wct = eda.coerencia_wavelet(
        base.loc[na_janela, clima], base.loc[na_janela, f"infl_sa_{grupo}"],
        mc=cfg["wavelet"]["mc_simulacoes"], semente=semente,
    )  # fmt: skip
    tabelas_wct.append(wct.assign(grupo=grupo, clima=clima))
coerencia = pd.concat(tabelas_wct, ignore_index=True)
coerencia.to_csv(TABELAS / "wavelet_coerencia.csv.gz", index=False)
faixas = pd.cut(
    coerencia["periodo"], [0, 6, 18, 64], labels=["2-6 meses", "6-18 meses", "18-64 meses"]
)
validas = coerencia[coerencia["dentro_coi"]].assign(faixa=faixas)
resumo_wct = (
    validas.groupby(["grupo", "clima", "faixa"], observed=True)
    .agg(coerencia_media=("coerencia", "mean"), pct_significativa=("significativa", "mean"))
    .reset_index()
)
resumo_wct["pct_significativa"] = (100 * resumo_wct["pct_significativa"]).round(1)
resumo_wct["coerencia_media"] = resumo_wct["coerencia_media"].round(3)
resumo_wct.to_csv(TABELAS / "wavelet_resumo.csv", index=False)
print(resumo_wct.to_string(index=False))
fig = visual.wavelet_paineis(coerencia, "Coerência wavelet clima × inflação dessazonalizada")
fig.savefig(FIGURAS / "wavelet_coerencia.png")
