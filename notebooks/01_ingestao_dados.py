# %% [markdown]
# # 01 — Ingestão de dados
#
# Baixa as séries do projeto, grava as respostas brutas em `data/raw/` (cache) e as tabelas
# processadas em `data/processed/` (Parquet) e `data/lakehouse.duckdb`.
#
# Rodar como script: `python notebooks/01_ingestao_dados.py [--refresh]`. Sem `--refresh`,
# tudo o que já está em cache é lido do disco.

# %%
import argparse

import pandas as pd

from clima_inflacao import lakehouse, visual
from clima_inflacao.config import caminho_projeto, carregar_config, garantir_diretorios
from clima_inflacao.ingestao import bcb, enso, inmet, ipca, malhas, nasa_power, producao
from clima_inflacao.ingestao.cliente_http import total_requisicoes
from clima_inflacao.ingestao.cobertura import cobertura
from clima_inflacao.log import configurar_log

parser = argparse.ArgumentParser(description="Parte 1: ingestão de dados")
parser.add_argument("--refresh", action="store_true", help="ignora o cache e baixa de novo")
args, _ = parser.parse_known_args()

config = carregar_config()
garantir_diretorios(config)
configurar_log(caminho_projeto("logs", config) / "01_ingestao.log")
TABELAS = caminho_projeto("tabelas", config)
BRUTOS = caminho_projeto("dados_brutos", config)
FIGURAS = caminho_projeto("figuras", config)
pd.set_option("display.width", 160, "display.max_columns", 20)

# %% [markdown]
# ## IPCA (IBGE/SIDRA)
#
# Cinco tabelas encadeadas (58, 655, 2938, 1419 e 7060), Brasil e 16 áreas. As categorias
# são identificadas pelo código estrutural do IPCA; o código SIDRA de cada tabela vem dos
# metadados e fica registrado em `outputs/tabelas/ipca_mapeamento_codigos.csv`.

# %%
dados_ipca, mapa_ipca = ipca.baixar_ipca(config, refresh=args.refresh)
dados_ipca = ipca.adicionar_indice(dados_ipca)
lakehouse.salvar(dados_ipca, "ipca", config)
lakehouse.salvar(mapa_ipca, "ipca_mapeamento_codigos", config)
mapa_ipca.to_csv(TABELAS / "ipca_mapeamento_codigos.csv", index=False)

print(mapa_ipca.pivot(index="categoria", columns="tabela", values="codigo_sidra"))

# %% [markdown]
# ### Cobertura por área e categoria
#
# Início, fim, nº de meses com variação e % de meses faltantes entre o início e o fim.

# %%
cobertura_ipca = cobertura(dados_ipca, ["area_id", "categoria"], coluna_valor="variacao")
cobertura_ipca.to_csv(TABELAS / "ipca_cobertura.csv", index=False)

geral = cobertura_ipca[cobertura_ipca["categoria"] == "indice_geral"]
print(geral[["area_id", "inicio", "fim", "n_obs", "pct_faltantes"]].to_string(index=False))
com_lacuna = cobertura_ipca[cobertura_ipca["pct_faltantes"] > 0]
print(f"Séries com meses faltantes entre início e fim: {len(com_lacuna)} de {len(cobertura_ipca)}")
print(com_lacuna.groupby("area_id")["categoria"].count().rename("n_series").to_string())

# %% [markdown]
# ### Quebras de série nas trocas de tabela (atualizações da POF)
#
# A variação mensal publicada encadeia as tabelas sem salto de nível, mas a composição e os
# pesos mudam em cada troca. A tabela abaixo mostra, para o Brasil, o nome publicado e o peso
# de cada categoria no último mês da tabela anterior e no primeiro da seguinte.

# %%
transicoes_ipca = ipca.transicoes(dados_ipca, mapa_ipca)
transicoes_ipca.to_csv(TABELAS / "ipca_transicoes_tabelas.csv", index=False)
print(transicoes_ipca.drop(columns=["tabela_anterior", "tabela_posterior"]).to_string(index=False))

# %% [markdown]
# ## Diesel: subitem "Óleo diesel" do IPCA
#
# Preferido à série da ANP. É índice de preço ao consumidor (variação mensal),
# não preço em R$/litro, e só existe a partir da tabela 655 (1999-08).

# %%
cfg_diesel = config["macro"]["diesel"]
diesel, mapa_diesel = ipca.baixar_ipca(
    config,
    refresh=args.refresh,
    categorias={"oleo_diesel": cfg_diesel["codigo_ipca"]},
    tabelas=cfg_diesel["tabelas"],
)
diesel = ipca.adicionar_indice(diesel)
lakehouse.salvar(diesel, "diesel_ipca", config)
mapa_diesel.to_csv(TABELAS / "diesel_mapeamento_codigos.csv", index=False)
cobertura_diesel = cobertura(diesel, ["area_id"], coluna_valor="variacao")
print(mapa_diesel[["tabela", "codigo_sidra", "nome_sidra"]].to_string(index=False))
print(cobertura_diesel.to_string(index=False))

# %% [markdown]
# ## Produção: PAM (anual, até microrregião)
#
# Lavouras temporárias (1612), permanentes (1613) e milho por safra (839), para Brasil, UFs e
# microrregiões. Consultas divididas em blocos de anos para respeitar o limite da API.

# %%
pam, unidades_pam = producao.baixar_pam(config, refresh=args.refresh)
lakehouse.salvar(pam, "pam", config)

mudancas = unidades_pam.groupby(["tabela", "produto", "variavel"])["unidade"].unique()
mudancas = mudancas[mudancas.map(len) > 1]
print("Variáveis com mais de uma unidade publicada:", "nenhuma" if mudancas.empty else "")
print(mudancas.to_string() if not mudancas.empty else "")

# Consistência: soma das microrregiões e das UFs contra o total do Brasil.
soma = pam.groupby(["ano", "produto", "nivel"])["quantidade_t"].sum(min_count=1).unstack("nivel")
for nivel in ["n3", "n9"]:
    dif = (soma[nivel] / soma["n1"] - 1).abs()
    print(f"Maior diferença relativa |soma {nivel} / Brasil - 1|: {dif.max():.2e}")

# Consistência: milho 1ª + 2ª safra (tabela 839) contra milho total (tabela 1612), Brasil.
brasil = pam[pam["nivel"] == "n1"].pivot(index="ano", columns="produto", values="quantidade_t")
safras = (brasil["milho_1a_safra"] + brasil["milho_2a_safra"]).dropna()
dif_milho = (safras / brasil.loc[safras.index, "milho"] - 1).abs()
print(f"Milho 1ª + 2ª safra contra total, maior diferença relativa: {dif_milho.max():.2e}")
print((brasil[["soja", "milho", "milho_1a_safra", "milho_2a_safra"]] / 1e6).tail(6).round(1))

# %% [markdown]
# ## Produção: LSPA (mensal, até UF)
#
# Em cada mês, a estimativa da safra do ano corrente; serve para acompanhar a safra dentro do
# ano, antes da PAM.

# %%
lspa = producao.baixar_lspa(config, refresh=args.refresh)
lakehouse.salvar(lspa, "lspa", config)
cobertura_lspa = cobertura(lspa[lspa["nivel"] == "n1"], ["produto"], coluna_valor="producao_t")
print(cobertura_lspa.to_string(index=False))

# %% [markdown]
# ## Macro: PTAX e IC-Br (BCB/SGS)
#
# PTAX diária (série 1) agregada em média mensal; IC-Br agropecuário (27575) e total (27574).
# A média mensal oficial (3698) serve só para validar a agregação.

# %%
cfg_bcb = config["macro"]["bcb_sgs"]
mensais = []
for nome, cfg_serie in cfg_bcb["series"].items():
    serie = bcb.serie_sgs(
        cfg_serie["codigo"],
        inicio=pd.Timestamp(config["macro"]["inicio"]),
        frequencia=cfg_serie["frequencia"],
        janela_max_anos=cfg_bcb["janela_max_anos_diaria"],
        dir_cache=BRUTOS / "bcb",
        http=config["http"],
        refresh=args.refresh,
    )
    if cfg_serie["frequencia"] == "diaria":
        lakehouse.salvar(serie, f"bcb_{nome}_diaria", config)
        mensal = bcb.media_mensal(serie)
    else:
        mensal = serie.rename(columns={"data": "mes"}).assign(n_dias=pd.NA)
    mensais.append(mensal.assign(serie=nome, codigo_sgs=cfg_serie["codigo"]))
macro = pd.concat(mensais, ignore_index=True)[["mes", "serie", "codigo_sgs", "valor", "n_dias"]]
lakehouse.salvar(macro, "macro_mensal", config)
print(cobertura(macro, ["serie"]).to_string(index=False))

largo_macro = macro.pivot(index="mes", columns="serie", values="valor")
comparacao = largo_macro[["ptax_venda", "ptax_venda_media_mensal"]].dropna()
dif_ptax = (comparacao["ptax_venda"] - comparacao["ptax_venda_media_mensal"]).abs()
print(
    f"PTAX: média mensal calculada contra a série 3698 em {len(comparacao)} meses; "
    f"maior diferença absoluta {dif_ptax.max():.5f} R$/US$ ({dif_ptax.idxmax():%Y-%m})"
)

# %% [markdown]
# ## ENSO: ONI e MEI v2 (NOAA PSL)

# %%
indices_enso = enso.baixar_enso(config, refresh=args.refresh)
lakehouse.salvar(indices_enso, "enso", config)
print(cobertura(indices_enso, ["indice"]).to_string(index=False))

# %% [markdown]
# ## Malhas e pontos de referência (geobr)
#
# Um ponto por microrregião (dentro do polígono) e a sede de cada área do IPCA. As
# microrregiões são a unidade dos pesos da PAM; o cinturão verde (Parte 3) também sai delas.

# %%
malha_micro = malhas.microrregioes(config, refresh=args.refresh)
pontos_micro = malhas.pontos_microrregioes(malha_micro)
pontos_areas = malhas.pontos_areas_ipca(
    config, malhas.sedes_municipais(config, refresh=args.refresh)
)
print(f"Microrregiões: {len(pontos_micro)}")
print(
    "Distância entre a sede do geobr e a coordenada preliminar do settings: "
    f"máx. {pontos_areas['dist_coord_settings_km'].max():.1f} km "
    f"({pontos_areas.loc[pontos_areas['dist_coord_settings_km'].idxmax(), 'nome']})"
)

# %% [markdown]
# ## INMET (complementar): estações automáticas perto das sedes
#
# Arquivos anuais do portal (a rota de dados da API exige token). O cache guarda as estações a
# até `distancia_max_km` de alguma sede; para cada área, fica a estação a até
# `distancia_escolha_km` com mais meses válidos de chuva.

# %%
cfg_inmet = config["clima"]["inmet"]
horario_inmet = pd.concat(
    [
        inmet.baixar_ano(
            ano, pontos_areas, cfg_inmet["distancia_max_km"], config, refresh=args.refresh
        )
        for ano in range(cfg_inmet["anos"][0], cfg_inmet["anos"][1] + 1)
    ],
    ignore_index=True,
)
# Coordenada mais recente de cada estação: a A104 (Rio Branco), por exemplo, aparece com
# longitude -68,165 nos arquivos antigos e -67,869 nos recentes.
estacoes_inmet = horario_inmet.drop_duplicates("codigo", keep="last")[
    ["codigo", "estacao", "uf", "lat_estacao", "lon_estacao"]
]
inmet_mensal = inmet.mensal(
    horario_inmet, cfg_inmet["fuso_horas"], cfg_inmet["min_pct_horas_validas"]
)
escolhidas = inmet.escolher_estacoes(
    inmet_mensal, estacoes_inmet, pontos_areas, cfg_inmet["distancia_escolha_km"]
)
lakehouse.salvar(inmet_mensal, "inmet_mensal", config)
lakehouse.salvar(escolhidas, "inmet_estacoes_escolhidas", config)
print(f"Estações a até {cfg_inmet['distancia_max_km']} km de alguma sede: {len(estacoes_inmet)}")
print(
    escolhidas[
        ["ponto_id", "codigo", "estacao", "distancia_km", "meses_validos", "n_candidatas"]
    ].to_string(index=False)
)

# %% [markdown]
# ## NASA POWER (base): clima diário por célula da grade
#
# Uma requisição por célula de 0,5° x 0,625° que contenha algum ponto: microrregiões, sedes
# das áreas e estações do INMET escolhidas (para a validação).

# %%
pontos_estacoes = pd.DataFrame(
    {
        "ponto_id": "inmet_" + escolhidas["codigo"],
        "tipo": "estacao_inmet",
        "codigo": escolhidas["codigo"],
        "nome": escolhidas["estacao"],
        "uf": escolhidas["uf"],
        "lat": escolhidas["lat_estacao"],
        "lon": escolhidas["lon_estacao"],
    }
)
pontos_clima = pd.concat([pontos_micro, pontos_areas, pontos_estacoes], ignore_index=True)
clima_diario, pontos_clima = nasa_power.baixar_clima(pontos_clima, config, refresh=args.refresh)
lakehouse.salvar(clima_diario, "clima_diario", config)
lakehouse.salvar(pontos_clima, "clima_pontos", config)
print(
    f"Pontos: {len(pontos_clima)}; células únicas: {pontos_clima['celula_id'].nunique()}; "
    f"linhas diárias: {len(clima_diario):,}"
)

# %% [markdown]
# ### Validação: NASA POWER contra o INMET
#
# Chuva e temperatura mensais da estação contra a célula do NASA POWER que a contém.

# %%
mapa_estacoes = escolhidas[["ponto_id", "codigo"]].merge(
    pontos_clima[pontos_clima["tipo"] == "estacao_inmet"][["codigo", "celula_id"]], on="codigo"
)
validacao = inmet.validar_contra_nasa(inmet_mensal, clima_diario, mapa_estacoes)
validacao.to_csv(TABELAS / "validacao_nasa_inmet.csv", index=False)
print(validacao.to_string(index=False))
print(validacao.drop(columns=["ponto_id", "codigo"]).describe().round(2))

# %% [markdown]
# ## Relatório de cobertura de todas as séries
#
# Início, fim, nº de observações e % de faltantes entre o início e o fim de cada série. A
# coluna `linha_figura` marca as linhas usadas na figura de cobertura.


# %%
def resumo(df, fonte, serie, coluna_valor, *, coluna_data="mes", freq="MS", figura=None, nota=""):
    """Cobertura de uma única série, com as colunas de identificação do relatório."""
    linha = cobertura(
        df.assign(_serie=serie),
        ["_serie"],
        coluna_data=coluna_data,
        coluna_valor=coluna_valor,
        freq=freq,
    )
    return linha.drop(columns="_serie").assign(
        fonte=fonte, serie=serie, frequencia=freq, linha_figura=figura, nota=nota
    )


linhas = []

# IPCA: as 10 categorias têm a mesma cobertura em cada área, então basta uma por área.
assert cobertura_ipca.groupby("area_id")[["inicio", "fim", "n_obs"]].nunique().eq(1).all().all()
ipca_alim = dados_ipca[dados_ipca["categoria"] == "alimentacao_domicilio"]
for area, grupo in ipca_alim.groupby("area_id"):
    linhas.append(resumo(grupo, "IPCA (SIDRA)", f"variação, {area}", "variacao"))
nomes_areas = {area["id"]: area["nome"] for area in config["ipca"]["areas"]["lista"]}
entrada = ipca_alim.groupby("area_id")["mes"].min()
for inicio, areas in entrada.groupby(entrada).groups.items():
    nomes = ", ".join(sorted(nomes_areas[a] for a in areas))
    rotulo = "IPCA: Brasil e 11 áreas" if len(areas) > 5 else f"IPCA: {nomes}"
    exemplo = ipca_alim[ipca_alim["area_id"] == areas[0]]
    linhas.append(
        resumo(exemplo, "IPCA (SIDRA)", f"áreas que entram em {inicio:%Y-%m}", "variacao",
               figura=rotulo, nota=f"{len(areas)} áreas; cobertura igual à de {areas[0]}")
    )  # fmt: skip
brasil_alim = ipca_alim[ipca_alim["area_id"] == "brasil"]
linhas.append(resumo(brasil_alim, "IPCA (SIDRA)", "peso, brasil", "peso", figura="IPCA: pesos"))
diesel_brasil = diesel[diesel["area_id"] == "brasil"]
linhas.append(
    resumo(diesel_brasil, "IPCA (SIDRA)", "óleo diesel, brasil", "variacao", figura="Diesel (IPCA)")
)

# PAM: total anual por tabela e nível territorial.
rotulos_pam = {1612: "PAM: temporárias", 1613: "PAM: permanentes", 839: "PAM: milho por safra"}
for (tabela, nivel), grupo in pam.groupby(["tabela", "nivel"]):
    total = grupo.groupby("ano", as_index=False)["quantidade_t"].sum(min_count=1)
    total["ano"] = pd.to_datetime(total["ano"].astype(str), format="%Y")
    figura_pam = rotulos_pam[tabela] if nivel == "n9" else None
    linhas.append(
        resumo(total, "PAM (SIDRA)", f"tabela {tabela}, {nivel}", "quantidade_t",
               coluna_data="ano", freq="YS", figura=figura_pam)
    )  # fmt: skip

# LSPA, BCB e ENSO.
for produto, grupo in lspa[lspa["nivel"] == "n1"].groupby("produto"):
    figura_lspa = "LSPA" if produto == "soja" else None
    linhas.append(
        resumo(grupo, "LSPA (SIDRA)", f"{produto}, brasil", "producao_t", figura=figura_lspa)
    )
rotulos_macro = {"ptax_venda": "PTAX", "icbr_agropecuaria": "IC-Br agropecuária"}
for serie, grupo in macro.groupby("serie"):
    linhas.append(resumo(grupo, "BCB (SGS)", serie, "valor", figura=rotulos_macro.get(serie)))
rotulos_enso = {"oni": "ONI", "mei_v2": "MEI v2"}
for indice, grupo in indices_enso.groupby("indice"):
    linhas.append(resumo(grupo, "NOAA PSL", indice, "valor", figura=rotulos_enso[indice]))

# NASA POWER: cobertura por célula, resumida por variável.
rotulos_clima = {
    "precipitacao_mm": "NASA POWER: chuva e temperatura",
    "radiacao_mj_m2": "NASA POWER: radiação",
}
for variavel in config["clima"]["nasa_power"]["parametros"]:
    por_celula = cobertura(
        clima_diario, ["celula_id"], coluna_data="data", coluna_valor=variavel, freq="D"
    )
    linhas.append(
        pd.DataFrame(
            {
                "inicio": [por_celula["inicio"].min()],
                "fim": [por_celula["fim"].max()],
                "n_obs": [int(por_celula["n_obs"].median())],
                "n_esperado": [int(por_celula["n_esperado"].median())],
                "pct_faltantes": [por_celula["pct_faltantes"].max()],
                "fonte": "NASA POWER",
                "serie": variavel,
                "frequencia": "D",
                "linha_figura": rotulos_clima.get(variavel),
                "nota": f"{len(por_celula)} células; n_obs e n_esperado são medianas; "
                "pct_faltantes é o máximo entre células",
            }
        )
    )

# INMET: chuva mensal das estações escolhidas.
for codigo, grupo in inmet_mensal[inmet_mensal["codigo"].isin(escolhidas["codigo"])].groupby(
    "codigo"
):
    figura_inmet = "INMET: estação de Brasília" if codigo == "A001" else None
    linhas.append(
        resumo(grupo, "INMET", f"chuva mensal, {codigo}", "precipitacao_mm", figura=figura_inmet)
    )

colunas = ["fonte", "serie", "frequencia", "inicio", "fim", "n_obs", "n_esperado",
           "pct_faltantes", "linha_figura", "nota"]  # fmt: skip
relatorio = pd.concat(linhas, ignore_index=True)[colunas]
relatorio.to_csv(TABELAS / "cobertura_series.csv", index=False)
print(relatorio.drop(columns=["linha_figura", "nota"]).to_string(index=False))

# %% [markdown]
# ## Janela amostral
#
# Definida e justificada no settings (`periodos.amostra_estimacao`).

# %%
amostra = config["periodos"]["amostra_estimacao"]
janela = (pd.Timestamp(amostra["inicio"]), pd.Timestamp(amostra["fim"]))
meses_janela = len(pd.date_range(*janela, freq="MS"))
print(f"Janela: {janela[0]:%Y-%m} a {janela[1]:%Y-%m} ({meses_janela} meses)")
na_janela = relatorio[(relatorio["frequencia"] == "MS") & relatorio["linha_figura"].notna()]
for linha in na_janela.itertuples():
    print(f"  {linha.linha_figura:32s} cobre a janela inteira: "
          f"{linha.inicio <= janela[0] and linha.fim >= janela[1]}")  # fmt: skip

ordem_figura = [
    "IPCA: Brasil e 11 áreas", "IPCA: Campo Grande, Vitória",
    "IPCA: Aracaju, Rio Branco, São Luís", "IPCA: pesos", "Diesel (IPCA)", "PAM: temporárias",
    "PAM: permanentes", "PAM: milho por safra", "LSPA", "PTAX", "IC-Br agropecuária", "ONI",
    "MEI v2", "NASA POWER: chuva e temperatura", "NASA POWER: radiação",
    "INMET: estação de Brasília",
]  # fmt: skip
figura = relatorio.dropna(subset=["linha_figura"]).set_index("linha_figura").loc[ordem_figura]
fig = visual.grafico_cobertura(
    figura.reset_index(),
    janela,
    inicio_eixo=pd.Timestamp("1981-01-01"),
    titulo=f"Cobertura das séries e janela amostral ({janela[0]:%Y-%m} a {janela[1]:%Y-%m}, "
    "faixa cinza)",
)
fig.savefig(FIGURAS / "cobertura_series.png")

# %%
print(f"Requisições HTTP nesta execução: {total_requisicoes()}")
