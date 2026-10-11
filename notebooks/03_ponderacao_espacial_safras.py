# %% [markdown]
# # 03 — Ponderação espacial das safras
#
# Pesos de produção por microrregião (PAM), índice climático ponderado pela safra (CWI) para o
# Brasil e para as regiões de monocultura (Centro-Oeste e Matopiba) e clima do cinturão verde
# de cada área do IPCA. O CWI é validado contra quebras de safra conhecidas, definidas no
# settings antes do cálculo.
#
# Rodar como script: `python notebooks/03_ponderacao_espacial_safras.py`.

# %%
import numpy as np
import pandas as pd

from clima_inflacao import lakehouse, visual
from clima_inflacao.config import caminho_projeto, carregar_config, garantir_diretorios
from clima_inflacao.espacial import cinturao_verde, cwi, pesos
from clima_inflacao.ingestao import malhas
from clima_inflacao.log import configurar_log

config = carregar_config()
garantir_diretorios(config)
configurar_log(caminho_projeto("logs", config) / "03_espacial.log")
TABELAS = caminho_projeto("tabelas", config)
FIGURAS = caminho_projeto("figuras", config)
cfg = config["espacial"]
amostra = config["periodos"]["amostra_estimacao"]
pd.set_option("display.width", 160, "display.max_columns", 30)

pam = lakehouse.ler("pam", config)
features = lakehouse.ler("features_clima_mensal", config)
pontos = lakehouse.ler("clima_pontos", config)
malha = malhas.microrregioes(config)

# %% [markdown]
# ## Pesos de produção por microrregião
#
# Principal: defasados (média de t-1 a t-3). Robustez: fixos (média de 2003-2007).

# %%
producao = pesos.producao_microrregioes(pam, cfg["pesos"]["culturas"])
anos = list(
    range(pd.Timestamp(amostra["inicio"]).year, pd.Timestamp(features["mes"].max()).year + 1)
)
pesos_def = pesos.pesos_defasados(producao, anos, tuple(cfg["pesos"]["anos_defasagem"]))
base = cfg["pesos"]["base_fixa"]
pesos_fix = pesos.pesos_fixos(producao, base["inicio"], base["fim"])

somas = pd.concat(
    [
        pesos_def.groupby(["cultura", "ano"])["peso"].sum(),
        pesos_fix.groupby("cultura")["peso"].sum(),
    ]
)
assert np.allclose(somas, 1.0), "pesos não somam 1"
print(f"Pesos somam 1 em todas as {len(somas)} combinações de cultura e ano (máx. desvio "
      f"{(somas - 1).abs().max():.1e})")  # fmt: skip
print(pesos_def.groupby(["cultura", "origem"])["ano"].nunique().unstack(fill_value=0))
lakehouse.salvar(pesos_def, "pesos_producao_defasados", config)
lakehouse.salvar(pesos_fix, "pesos_producao_fixos", config)

# Regiões de monocultura.
micro_uf = pontos[pontos["tipo"] == "microrregiao"][["codigo", "uf"]]
codigos_regiao = {
    "brasil": micro_uf["codigo"].tolist(),
    "centro_oeste": micro_uf[
        micro_uf["uf"].isin(cfg["regioes_monocultura"]["centro_oeste"]["ufs"])
    ]["codigo"].tolist(),
    "matopiba": [str(c) for c in cfg["regioes_monocultura"]["matopiba"]["microrregioes"]],
}
assert set(codigos_regiao["matopiba"]) <= set(codigos_regiao["brasil"])
ano_ref = pd.Timestamp(amostra["fim"]).year
participacao = (
    pd.DataFrame(
        {
            regiao: pesos_def[(pesos_def["ano"] == ano_ref) & pesos_def["codigo"].isin(codigos)]
            .groupby("cultura")["peso"]
            .sum()
            for regiao, codigos in codigos_regiao.items()
            if regiao != "brasil"
        }
    )
    .mul(100)
    .round(1)
)
participacao.to_csv(TABELAS / "participacao_regioes_monocultura.csv")
janela_ref = f"{ano_ref - 3}-{ano_ref - 1}"
print(f"Participação (%) na produção nacional, pesos de {ano_ref} (média {janela_ref}):")
print(participacao.to_string())

# %% [markdown]
# ## CWI: SPI ponderado pela produção
#
# CWI(t) = soma_i w(i,k) x SPI(i,t), por cultura, região e versão dos pesos.

# %%
colunas = cfg["cwi"]["indicadores"]
spi_micro = pontos[pontos["tipo"] == "microrregiao"][["codigo", "celula_id"]].merge(
    features[["celula_id", "mes", *colunas]], on="celula_id"
)
spi_micro = spi_micro[spi_micro["mes"] >= pd.Timestamp(f"{anos[0]}-01-01")].drop(
    columns="celula_id"
)
partes = []
for versao, tabela_pesos in [("defasado", pesos_def), ("fixo", pesos_fix)]:
    for regiao, codigos in codigos_regiao.items():
        p = pesos.restringir(tabela_pesos, codigos)
        partes.append(cwi.cwi(spi_micro, p, colunas).assign(versao=versao, regiao=regiao))
cwi_mensal = pd.concat(partes, ignore_index=True)
lakehouse.salvar(cwi_mensal, "cwi_mensal", config)
cobertura_min = cwi_mensal[[f"cobertura_{c}" for c in colunas]].min().round(4).to_dict()
print(f"CWI: {len(cwi_mensal):,} linhas; cobertura mínima dos pesos com dado: {cobertura_min}")

principal = cwi_mensal[(cwi_mensal["versao"] == "defasado") & (cwi_mensal["mes"] <= amostra["fim"])]
correlacoes = (
    principal[principal["cultura"].isin(["soja", "milho_2a_safra"])]
    .pivot_table(index=["cultura", "mes"], columns="regiao", values="spi_6")
    .groupby("cultura")
    .corr()
    .round(2)
)
print("Correlação do CWI (SPI-6) entre regiões, na janela de estimação:")
print(correlacoes)
comparacao = cwi_mensal[cwi_mensal["regiao"] == "brasil"].pivot_table(
    index=["cultura", "mes"], columns="versao", values="spi_6"
)
corr_versoes = comparacao.groupby("cultura").corr().xs("defasado", level=1)["fixo"].round(3)
print("Correlação entre pesos defasados e fixos (Brasil, SPI-6):", corr_versoes.to_dict())

# %% [markdown]
# ## Critério de aceite: o CWI se move junto com quebras de safra conhecidas?
#
# Definido no settings antes do cálculo (`espacial.validacao_cwi`): CWI nacional negativo em
# cada ano de quebra e correlação positiva (p < 0,05) com a anomalia do rendimento nacional.

# %%
brasil_principal = cwi_mensal[
    (cwi_mensal["versao"] == "defasado") & (cwi_mensal["regiao"] == "brasil")
]
rendimento = pam[pam["nivel"] == "n1"][["ano", "produto", "rendimento_kg_ha"]]
anos_validacao = (anos[0], pd.Timestamp(amostra["fim"]).year)
resumo_cwi, tabela_cwi = cwi.validar(
    brasil_principal, rendimento, cfg["validacao_cwi"], anos_validacao, coluna="spi_3"
)
resumo_cwi.to_csv(TABELAS / "validacao_cwi.csv", index=False)
tabela_cwi.to_csv(TABELAS / "cwi_rendimento_safras.csv", index=False)
print(resumo_cwi.to_string(index=False))
fig = visual.dispersao_cwi_rendimento(
    tabela_cwi, resumo_cwi, "CWI no fim da janela crítica contra a anomalia do rendimento nacional"
)
fig.savefig(FIGURAS / "cwi_rendimento_safras.png")

# %% [markdown]
# ## Mapas de checagem dos pesos

# %%
pesos_mapa = pesos_def[pesos_def["ano"] == ano_ref]
pesos_mapa[pesos_mapa["cultura"].isin(["soja", "milho_1a_safra", "milho_2a_safra"])].to_csv(
    TABELAS / "pesos_mapa.csv", index=False
)
malha_cod = malha.assign(codigo=malha["code_micro"].astype(str))
contornos = {
    "Centro-Oeste": malha_cod[malha_cod["codigo"].isin(codigos_regiao["centro_oeste"])].dissolve(),
    "Matopiba": malha_cod[malha_cod["codigo"].isin(codigos_regiao["matopiba"])].dissolve(),
}
paineis = [
    (rotulo, pesos_mapa[pesos_mapa["cultura"] == cultura][["codigo", "peso"]])
    for cultura, rotulo in [("soja", "Soja"), ("milho_1a_safra", "Milho 1ª safra"),
                            ("milho_2a_safra", "Milho 2ª safra")]
]  # fmt: skip
fig = visual.mapas_pesos(
    malha, paineis, contornos,
    f"Participação de cada microrregião na produção nacional (pesos de {ano_ref})",
)  # fmt: skip
fig.savefig(FIGURAS / "pesos_producao_mapas.png")

# %% [markdown]
# ## Cinturão verde: clima num raio em torno da sede de cada área
#
# Média simples das células do NASA POWER com centro a até 150 e 300 km da sede.

# %%
sedes = pontos[pontos["tipo"] == "area_ipca"].assign(
    area_id=lambda d: d["ponto_id"].str.removeprefix("area_")
)
celulas = pontos.drop_duplicates("celula_id")[["celula_id", "lat_celula", "lon_celula"]]
selecao = cinturao_verde.celulas_no_raio(sedes, celulas, cfg["raios_cinturao_verde_km"])
selecao.to_csv(TABELAS / "cinturao_verde_celulas.csv", index=False)
contagem = selecao.groupby(["area_id", "raio_km"])["celula_id"].nunique().unstack()
print("Células por área e raio:")
print(contagem.T.to_string())
cinturao = cinturao_verde.agregar(features, selecao, cfg["cinturao_verde"]["variaveis"])
lakehouse.salvar(cinturao, "cinturao_verde_mensal", config)
fig = visual.mapa_cinturoes(
    malha, sedes, cfg["raios_cinturao_verde_km"],
    "Cinturões verdes: 150 km (linha cheia) e 300 km (tracejada) em torno das sedes",
    rotulos_a_esquerda=("Goiânia", "Belo Horizonte", "São Paulo", "Salvador", "Campo Grande"),
)  # fmt: skip
fig.savefig(FIGURAS / "cinturao_verde_mapa.png")

# Densidade: só há células onde existe algum ponto (microrregião, sede ou estação), então o
# raio de 150 km tem poucas células em áreas de microrregiões grandes. Se os agregados de 150 e
# 300 km andam juntos, a amostragem esparsa pesa pouco.
largo = cinturao[cinturao["mes"] <= amostra["fim"]].pivot_table(
    index=["area_id", "mes"], columns="raio_km", values="spi_3"
)
densidade = pd.DataFrame(
    {
        "celulas_150km": contagem[150],
        "celulas_300km": contagem[300],
        "corr_spi3_150_300": largo.groupby("area_id").apply(lambda g: g[150].corr(g[300])).round(3),
    }
)
densidade.to_csv(TABELAS / "cinturao_verde_densidade.csv")
print(densidade.sort_values("corr_spi3_150_300").to_string())
