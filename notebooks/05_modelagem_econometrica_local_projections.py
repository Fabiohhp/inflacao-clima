# %% [markdown]
# # 05 — Local Projections
#
# **Parte 5 (agregada):** resposta acumulada do IPCA de alimentação no domicílio (Brasil) a um
# choque de seca de 1 unidade de SPI (choque = −SPI), h = 0 a 18 meses (Jordà, 2005), com erros
# HAC de Newey-West (h + 1 lags) e bandas de 68% e 90%. Especificação registrada no settings
# antes da estimação (`modelos.local_projections`).
#
# **Parte 6a (por grupo e não linear):** a mesma especificação por grupo do IPCA, a resposta
# marginal (mês de maior aceleração do preço) e a versão com seca, excesso de chuva e calor
# medidos como fração da produção em condição extrema (`setorial` e `nao_linear`, registrados
# antes de estimar). Ressalvas adicionais: R23 a R25.
#
# **Parte 6b (por área do IPCA):** em cada uma das 11 áreas com IPCA desde a janela, a resposta
# ao clima do cinturão verde da própria área e ao das regiões de monocultura (Centro-Oeste e
# Matopiba), no mesmo modelo, e painéis das 11 áreas, um deles com efeito fixo de mês, que
# isola o efeito do clima local de todo choque comum (`areas`, registrado antes de estimar).
# Ressalvas centrais: R2, R4, R5, R8.
#
# **Parte 6c (ENSO × anomalias locais):** quanto o ENSO explica do clima local, a resposta dos
# grupos ao ENSO, o controle do canal da ração por ENSO (R21) e a previsão pseudo fora da
# amostra (janela expansiva, 2012-01 a 2023-06) com o ENSO, o clima local e os dois juntos
# (`enso` e `modelos.previsao`, registrados antes de estimar). Ressalvas centrais: R12, R21.
#
# Ressalvas que se aplicam a todos os resultados: R1 (janela até 2023-06), R2 e R5 (erro de
# medida no clima → atenuação), R3 (CWI da soja com ruído), R16 (tendência do SPI → tendência
# linear na regressão), R17 (efeitos não lineares ficam para a Parte 6) e R20 (agregação
# nacional). Ver docs/RESSALVAS.md.
#
# Rodar como script: `python notebooks/05_modelagem_econometrica_local_projections.py`.

# %%
import os

# BLAS com 1 thread: os painéis da 6b (matrizes de ~3.000 x 430) rodam ~5x mais rápido sem a
# disputa entre threads. Precisa vir antes de importar o numpy.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from clima_inflacao import lakehouse, visual  # noqa: E402
from clima_inflacao.config import (  # noqa: E402
    caminho_projeto,
    carregar_config,
    garantir_diretorios,
)
from clima_inflacao.espacial import cinturao_verde, cwi  # noqa: E402
from clima_inflacao.features import extremos  # noqa: E402
from clima_inflacao.log import configurar_log  # noqa: E402
from clima_inflacao.modelos import local_projections as lp  # noqa: E402
from clima_inflacao.modelos import previsao as pv  # noqa: E402

config = carregar_config()
garantir_diretorios(config)
configurar_log(caminho_projeto("logs", config) / "05_lp.log")
TABELAS = caminho_projeto("tabelas", config)
FIGURAS = caminho_projeto("figuras", config)
cfg = config["modelos"]["local_projections"]
ag = cfg["agregada"]
INICIO = pd.Timestamp(cfg["inicio_efetivo"])
FIM = pd.Timestamp(config["periodos"]["amostra_estimacao"]["fim"])
HORIZONTES = range(cfg["horizonte_min"], cfg["horizonte_max"] + 1)
pd.set_option("display.width", 170, "display.max_columns", 30)

base = lakehouse.ler("base_analise_brasil", config).set_index("mes")
choques = sorted(set(ag["choques"]) | set(ag["conjunto"]))
for choque in choques:
    base[f"seca_{choque}"] = cfg["sinal_choque"] * base[choque]
ROTULOS = {
    "seca_cinturao_spi_3": "Seca no cinturão verde (SPI-3)",
    "seca_cwi_milho_spi_6": "Seca no milho (CWI, SPI-6)",
    "seca_cwi_soja_spi_6": "Seca na soja (CWI, SPI-6)",
}
NOMES_ESPEC = {
    "principal": "principal (dummies de mês + tendência)",
    "com_controles": "+ câmbio, IC-Br e diesel",
    "dessazonalizada": "série dessazonalizada (STL)",
    "sem_tendencia": "sem tendência",
}

# %% [markdown]
# ## Especificações

# %%
dependente = ag["dependente"]


def especificacao(
    nome: str, choques_modelo: list[str], dependente: str = dependente, **extra
) -> lp.EspecificacaoLP:
    """Especificação `nome` para o log-preço de `dependente`; `extra` vai para EspecificacaoLP
    (ex.: acumulada=False, contrastes)."""
    comum = {
        "choques": choques_modelo,
        "horizontes": HORIZONTES,
        "lags_dependente": cfg["lags_dependente"],
        "lags_choque": cfg["lags_choque"],
        "lags_controles": cfg["lags_controles"],
        "hac_lags_extra": cfg["hac_lags_extra"],
        "niveis_confianca": tuple(cfg["niveis_confianca"]),
        **extra,
    }
    if nome == "principal":
        return lp.EspecificacaoLP(log_preco=f"log_{dependente}", **comum)
    if nome == "com_controles":
        return lp.EspecificacaoLP(
            log_preco=f"log_{dependente}", controles=cfg["controles_macro"], **comum
        )
    if nome == "dessazonalizada":
        return lp.EspecificacaoLP(log_preco=f"log_sa_{dependente}", dummies_mes=False, **comum)
    if nome == "sem_tendencia":
        return lp.EspecificacaoLP(log_preco=f"log_{dependente}", tendencia=False, **comum)
    raise ValueError(nome)


resultados = []
for choque in ag["choques"]:
    for nome in ag["especificacoes"]:
        irf = lp.estimar(base, especificacao(nome, [f"seca_{choque}"]), INICIO, FIM)
        resultados.append(irf.assign(especificacao=nome, dependente=dependente))
conjunto = lp.estimar(
    base, especificacao("principal", [f"seca_{c}" for c in ag["conjunto"]]), INICIO, FIM
)
resultados.append(conjunto.assign(especificacao="conjunto", dependente=dependente))
irfs = pd.concat(resultados, ignore_index=True)

# Escala: 1 unidade de SPI não é 1 desvio-padrão do choque. O CWI é média ponderada de muitas
# microrregiões e varia menos que o SPI de cada uma; a resposta também sai por 1 desvio-padrão
# amostral de cada choque (colunas *_dp).
na_amostra = (base.index >= INICIO) & (base.index <= FIM)
desvios = {f"seca_{c}": base.loc[na_amostra, f"seca_{c}"].std() for c in choques}
irfs["dp_choque"] = irfs["choque"].map(desvios)
for coluna in ["coef", "li_68", "ls_68", "li_90", "ls_90"]:
    irfs[f"{coluna}_dp"] = irfs[coluna] * irfs["dp_choque"]
print("Desvio-padrão amostral de cada choque:", {k: round(v, 3) for k, v in desvios.items()})
irfs.to_csv(TABELAS / "lp_agregada.csv", index=False)
lakehouse.salvar(irfs, "lp_agregada", config)
print(f"Amostra: t de {INICIO:%Y-%m} a {FIM:%Y-%m} − h; n de {irfs['n'].max()} (h = 0) a "
      f"{irfs['n'].min()} (h = {cfg['horizonte_max']})")  # fmt: skip

# %% [markdown]
# ## Resultados
#
# Coeficientes em pontos percentuais de variação acumulada do preço por 1 unidade de SPI de
# seca (e, nas colunas *_dp, por 1 desvio-padrão amostral do choque). "horizontes_signif_90"
# conta os h cuja banda de 90% exclui zero, sem correção para testes múltiplos (19 horizontes
# correlacionados).

# %%
resumo = lp.resumo_irf(irfs, nivel=90)
resumo.to_csv(TABELAS / "lp_agregada_resumo.csv", index=False)
print(resumo.round(3).to_string(index=False))

principal = irfs[irfs["especificacao"] == "principal"]
colunas = ["choque", "horizonte", "coef", "li_90", "ls_90", "coef_dp", "p_valor", "n"]
selecao = principal[principal["horizonte"].isin([0, 3, 6, 9, 12, 15, 18])][colunas]
print(selecao.round(3).to_string(index=False))

dp = {k: visual.numero(v) for k, v in desvios.items()}
nota = (
    "Resposta acumulada do IPCA de alimentação no domicílio (Brasil) a 1 unidade de SPI de seca "
    f"(−SPI) em t; desvio-padrão amostral do choque: cinturão {dp['seca_cinturao_spi_3']}, "
    f"milho {dp['seca_cwi_milho_spi_6']}, soja {dp['seca_cwi_soja_spi_6']}. Faixas: bandas HAC de "
    "68% e 90%. Amostra: 2000-09 a 2023-06. 12 defasagens do choque e da inflação, dummies de mês "
    "e tendência."
)
fig = visual.irf_paineis(
    principal, [f"seca_{c}" for c in ag["choques"]], ROTULOS,
    "Local Projections: alimentação no domicílio após uma seca", nota,
)  # fmt: skip
fig.savefig(FIGURAS / "lp_agregada_principal.png")
fig = visual.irf_comparacao(
    irfs, [f"seca_{c}" for c in ag["choques"]], ag["especificacoes"], ROTULOS, NOMES_ESPEC,
    "Local Projections: sensibilidade à especificação (estimativas pontuais)",
)  # fmt: skip
fig.savefig(FIGURAS / "lp_agregada_especificacoes.png")


# %% [markdown]
# ## Parte 6a — Local Projections por grupo
#
# Pares grupo × choque, especificações e regra de avaliação registrados antes de estimar
# (`modelos.local_projections.setorial`). O 1º choque de cada grupo é o motor principal. Para
# cada par: resposta acumulada (principal e com controles) e resposta marginal (principal),
# que dá o mês de maior aceleração do preço ("lag ótimo").

# %%
st = cfg["setorial"]
hipoteses = {g: tuple(j) for g, j in config["hipoteses_defasagens_meses"].items()}
choques_setor = sorted({c for lista in st["pares"].values() for c in lista})
for choque in choques_setor:
    base[f"seca_{choque}"] = cfg["sinal_choque"] * base[choque]
desvios |= {f"seca_{c}": base.loc[na_amostra, f"seca_{c}"].std() for c in choques_setor}

partes_setor = []
for grupo, choques_grupo in st["pares"].items():
    for choque in choques_grupo:
        modelo = [f"seca_{choque}"]
        rotulo = {"dependente": grupo, "motor_principal": choque == choques_grupo[0]}
        for nome in st["especificacoes"]:
            irf = lp.estimar(base, especificacao(nome, modelo, grupo), INICIO, FIM)
            partes_setor.append(irf.assign(especificacao=nome, tipo="acumulada", **rotulo))
        espec_marginal = especificacao("principal", modelo, grupo, acumulada=False)
        irf = lp.estimar(base, espec_marginal, INICIO, FIM)
        partes_setor.append(irf.assign(especificacao="principal", tipo="marginal", **rotulo))
setorial = pd.concat(partes_setor, ignore_index=True)
setorial["dp_choque"] = setorial["choque"].map(desvios)
for coluna in ["coef", "li_68", "ls_68", "li_90", "ls_90"]:
    setorial[f"{coluna}_dp"] = setorial[coluna] * setorial["dp_choque"]
setorial.to_csv(TABELAS / "lp_setorial.csv", index=False)
lakehouse.salvar(setorial, "lp_setorial", config)
n_respostas = setorial.groupby(["dependente", "choque", "especificacao", "tipo"]).ngroups
print(f"{n_respostas} respostas × {len(HORIZONTES)} horizontes")

# %% [markdown]
# ### Defasagens contra as hipóteses e dissipação
#
# Regra (settings): h_inicio = 1º horizonte com a banda de 90% da resposta acumulada acima de
# zero; lag_otimo = maior coeficiente da resposta marginal; dissipação = 1º horizonte após o
# pico em que a resposta acumulada cai abaixo de metade do pico. Coeficientes por 1
# desvio-padrão do choque (R22).


# %%
def _irf(dep, choque, espec, tipo):
    filtro = (
        (setorial["dependente"] == dep) & (setorial["choque"] == choque)
        & (setorial["especificacao"] == espec) & (setorial["tipo"] == tipo)
    )  # fmt: skip
    return setorial[filtro]


linhas = []
for grupo, choques_grupo in st["pares"].items():
    for choque in choques_grupo:
        nome = f"seca_{choque}"
        acum = _irf(grupo, nome, "principal", "acumulada")
        aval = lp.avaliar_defasagens(
            acum, _irf(grupo, nome, "principal", "marginal"), hipoteses.get(grupo),
            fracao_dissipacao=st["fracao_dissipacao"],
        )  # fmt: skip
        controles = _irf(grupo, nome, "com_controles", "acumulada").set_index("horizonte")
        dp = desvios[nome]
        linhas.append(
            {"dependente": grupo, "choque": choque, "motor_principal": choque == choques_grupo[0],
             "janela_hipotese": "–".join(map(str, hipoteses[grupo])) if grupo in hipoteses
             else "", **aval, "dp_choque": dp,
             "pico_coef_dp": aval["pico_coef"] * dp,
             "lag_otimo_coef_dp": aval["lag_otimo_coef"] * dp,
             "h12_dp": acum.set_index("horizonte").loc[12, "coef"] * dp,
             "h12_dp_com_controles": controles.loc[12, "coef"] * dp,
             "horizontes_acima_com_controles": int((controles["li_90"] > 0).sum()),
             "horizontes_abaixo_com_controles": int((controles["ls_90"] < 0).sum())}
        )  # fmt: skip
avaliacao = pd.DataFrame(linhas)
avaliacao.to_csv(TABELAS / "lp_setorial_avaliacao.csv", index=False)
colunas = ["dependente", "choque", "janela_hipotese", "h_inicio", "avaliacao", "lag_otimo",
           "horizontes_marginal_acima", "h_pico", "pico_coef_dp", "pico_signif", "h_dissipacao",
           "h12_dp", "h12_dp_com_controles", "horizontes_acima", "horizontes_acima_com_controles",
           "horizontes_abaixo"]  # fmt: skip
print(avaliacao[colunas].round(2).to_string(index=False))

# %% [markdown]
# ### Figuras por grupo
#
# Eixo y próprio em cada painel (a volatilidade dos grupos difere muito). Faixa cinza: janela
# de defasagem da hipótese do grupo.

# %%
NOMES_CHOQUES = {
    "seca_cinturao_spi_3": "cinturão verde (SPI-3)",
    "seca_cwi_milho_spi_6": "milho (CWI SPI-6)",
    "seca_cwi_milho_2a_safra_spi_6": "milho 2ª safra (CWI SPI-6)",
    "seca_cwi_soja_spi_6": "soja (CWI SPI-6)",
    "seca_cwi_batata_inglesa_spi_3": "batata (CWI SPI-3)",
    "seca_cwi_banana_spi_3": "banana (CWI SPI-3)",
    "seca_cwi_laranja_spi_6": "laranja (CWI SPI-6)",
    "seca_cwi_feijao_spi_3": "feijão (CWI SPI-3)",
    "seca_cwi_arroz_spi_3": "arroz (CWI SPI-3)",
}
titulos = {
    (g, f"seca_{c}"): f"{visual.NOMES_GRUPOS[g]}\nseca: {NOMES_CHOQUES[f'seca_{c}']}"
    for g, cs in st["pares"].items() for c in cs
}  # fmt: skip
principais = [(g, f"seca_{cs[0]}") for g, cs in st["pares"].items()]
acumulada = setorial[(setorial["tipo"] == "acumulada") & (setorial["especificacao"] == "principal")]
com_controles = setorial[setorial["especificacao"] == "com_controles"]
nota_grupos = (
    "Resposta por 1 desvio-padrão amostral do choque de seca (−SPI). Linha tracejada: estimativa "
    "com câmbio, IC-Br e diesel (R21). Amostra: t de 2000-09 a 2023-06 − h; 12 defasagens do "
    "choque e da inflação, dummies de mês e tendência; bandas HAC de Newey-West."
)
fig = visual.irf_grade(
    acumulada, principais, titulos, "Resposta acumulada de cada grupo a uma seca no seu motor "
    "climático principal", nota_grupos, janelas=hipoteses, comparacao=com_controles,
    nome_comparacao="com controles",
)  # fmt: skip
fig.savefig(FIGURAS / "lp_setorial_grupos.png")
marginal = setorial[setorial["tipo"] == "marginal"]
fig = visual.irf_grade(
    marginal, principais, titulos, "Resposta marginal: aceleração da inflação de cada grupo "
    "h meses após a seca", nota_grupos.replace(" Linha tracejada: estimativa com câmbio, IC-Br e "
    "diesel (R21).", " Coeficiente = efeito na inflação do mês t + h."), janelas=hipoteses,
    rotulo_y="inflação do mês t + h (p.p.)",
)  # fmt: skip
fig.savefig(FIGURAS / "lp_setorial_marginal.png")
proteina = [(g, f"seca_{c}") for g in ["carnes", "aves_ovos", "leite_derivados"]
            for c in st["pares"][g]]  # fmt: skip
fig = visual.irf_grade(
    acumulada, proteina, titulos, "Proteína animal: resposta a secas nas culturas de ração",
    nota_grupos, ncols=3, janelas=hipoteses, comparacao=com_controles,
    nome_comparacao="com controles",
)  # fmt: skip
fig.savefig(FIGURAS / "lp_setorial_proteina.png")

# %% [markdown]
# ## Parte 6a — Versão não linear: seca, excesso de chuva e calor
#
# O limiar vale em cada microrregião (CWI) ou célula (cinturão), e o choque é a fração da
# produção (ou das células do cinturão) em seca (SPI < −1,5), excesso (SPI > 1,5) ou calor
# (anomalia padronizada da Tmax média > 1,5). Os três entram juntos no mesmo modelo.

# %%
nl = cfg["nao_linear"]
limiares = config["choques"]
referencia = config["features"]["spi"]["periodo_referencia"]
TIPOS = ["seca", "excesso", "calor"]
features = lakehouse.ler("features_clima_mensal", config)
pontos = lakehouse.ler("clima_pontos", config)
pesos_def = lakehouse.ler("pesos_producao_defasados", config)
indicadores = {
    spi: extremos.indicadores_limiar(
        features, spi, limiares["spi_seca"], limiares["spi_excesso"], nl["limiar_calor_z"],
        (referencia["inicio"], referencia["fim"]),
    )
    for spi in sorted({m["spi"] for m in nl["modelos"]})
}  # fmt: skip

# CWI das frações: mesma ponderação (pesos defasados, Brasil) do CWI da Parte 3.
micro = pontos[pontos["tipo"] == "microrregiao"][["codigo", "celula_id"]]
# Cinturão: mesmas células (150 km) e as 11 áreas com IPCA desde a janela, como na Parte 4.
ipca = lakehouse.ler("ipca", config)
entrada = ipca.groupby("area_id")["mes"].min()
inicio_janela = pd.Timestamp(config["periodos"]["amostra_estimacao"]["inicio"])
areas_longas = [a for a in entrada.index if entrada[a] <= inicio_janela and a != "brasil"]
sedes = pontos[pontos["tipo"] == "area_ipca"].assign(
    area_id=lambda d: d["ponto_id"].str.removeprefix("area_")
)
sedes = sedes[sedes["area_id"].isin(areas_longas)]
celulas = pontos.drop_duplicates("celula_id")[["celula_id", "lat_celula", "lon_celula"]]
selecao = cinturao_verde.celulas_no_raio(sedes, celulas, [config["eda"]["raio_cinturao_km"]])

fracoes = []
for spi, ind in indicadores.items():
    motores = {m["motor"] for m in nl["modelos"] if m["spi"] == spi}
    culturas = sorted(motores - {"cinturao"})
    por_micro = micro.merge(ind, on="celula_id").drop(columns="celula_id")
    agregado = cwi.cwi(por_micro, pesos_def[pesos_def["cultura"].isin(culturas)], TIPOS)
    fracoes.append(agregado.rename(columns={"cultura": "motor"})[["motor", "mes", *TIPOS]]
                   .assign(spi=spi))  # fmt: skip
    if "cinturao" in motores:
        por_area = cinturao_verde.agregar(ind, selecao, TIPOS)
        media = por_area.groupby("mes")[TIPOS].mean().reset_index()
        fracoes.append(media.assign(motor="cinturao", spi=spi))
fracoes = pd.concat(fracoes, ignore_index=True)
lakehouse.salvar(fracoes, "fracoes_extremos_mensal", config)
largo = fracoes.pivot_table(index="mes", columns=["motor", "spi"], values=TIPOS)
for tipo, motor, spi in largo.columns:
    base[f"fracao_{tipo}_{motor}_{spi}"] = largo[(tipo, motor, spi)]

# Distribuição dos choques na amostra, por metade (R23), e correlação entre os tipos.
resumo_fracoes = []
na_janela = fracoes[(fracoes["mes"] >= INICIO) & (fracoes["mes"] <= FIM)]
for (motor, spi), grupo in na_janela.groupby(["motor", "spi"]):
    metade = grupo["mes"].dt.year >= 2012
    corr = grupo[TIPOS].corr()
    for tipo in TIPOS:
        resumo_fracoes.append(
            {"motor": motor, "spi": spi, "tipo": tipo, "media": grupo[tipo].mean(),
             "desvio_padrao": grupo[tipo].std(), "maximo": grupo[tipo].max(),
             "media_2000_2011": grupo.loc[~metade, tipo].mean(),
             "media_2012_2023": grupo.loc[metade, tipo].mean(),
             "corr_seca": corr.loc[tipo, "seca"], "corr_excesso": corr.loc[tipo, "excesso"],
             "corr_calor": corr.loc[tipo, "calor"]}
        )  # fmt: skip
resumo_fracoes = pd.DataFrame(resumo_fracoes)
resumo_fracoes.to_csv(TABELAS / "fracoes_extremos_resumo.csv", index=False)
print(resumo_fracoes.round(3).to_string(index=False))

# %% [markdown]
# ### Estimação não linear e testes de assimetria
#
# Respostas por 10 p.p. de fração (colunas *_esc) e por 1 desvio-padrão (colunas *_dp). Os
# contrastes seca − excesso e calor − excesso usam a mesma matriz HAC; o teste de cada modelo
# é lido só no horizonte h_teste fixado no settings (18 testes no total).

# %%
partes_nl = []
for modelo in nl["modelos"]:
    dep, motor, spi = modelo["dependente"], modelo["motor"], modelo["spi"]
    nomes = {tipo: f"fracao_{tipo}_{motor}_{spi}" for tipo in TIPOS}
    contrastes = [(nomes[a], nomes[b]) for a, b in nl["contrastes"]]
    for nome in nl["especificacoes"]:
        espec = especificacao(nome, list(nomes.values()), dep, contrastes=contrastes)
        irf = lp.estimar(base, espec, INICIO, FIM)
        tipo_choque = {v: k for k, v in nomes.items()} | {
            f"{nomes[a]} − {nomes[b]}": f"{a} − {b}" for a, b in nl["contrastes"]
        }
        partes_nl.append(
            irf.assign(dependente=dep, motor=motor, spi=spi, especificacao=nome,
                       tipo_choque=irf["choque"].map(tipo_choque), h_teste=modelo["h_teste"])
        )  # fmt: skip
nao_linear = pd.concat(partes_nl, ignore_index=True)
for coluna in ["coef", "li_68", "ls_68", "li_90", "ls_90"]:
    nao_linear[f"{coluna}_esc"] = nao_linear[coluna] * nl["escala_fracao"]
desvio_fracao = {
    f"fracao_{tipo}_{motor}_{spi}": base.loc[na_amostra, f"fracao_{tipo}_{motor}_{spi}"].std()
    for tipo in TIPOS for motor, spi in sorted({(m["motor"], m["spi"]) for m in nl["modelos"]})
}  # fmt: skip
nao_linear["dp_choque"] = nao_linear["choque"].map(desvio_fracao)
nao_linear["coef_dp"] = nao_linear["coef"] * nao_linear["dp_choque"]
nao_linear.to_csv(TABELAS / "lp_nao_linear.csv", index=False)
lakehouse.salvar(nao_linear, "lp_nao_linear", config)

no_teste = nao_linear[nao_linear["horizonte"] == nao_linear["h_teste"]].copy()
no_teste["modelo"] = no_teste["dependente"] + "|" + no_teste["motor"]
colunas = ["especificacao", "dependente", "motor", "h_teste", "tipo_choque", "coef_esc",
           "li_90_esc", "ls_90_esc", "p_valor", "coef_dp", "n"]  # fmt: skip
teste = no_teste[colunas + ["modelo"]]
teste.to_csv(TABELAS / "lp_nao_linear_teste.csv", index=False)
print(teste[colunas].round(3).to_string(index=False))
# Resumo de cada resposta e dissipação (mesma regra da versão linear: 1º horizonte após o
# pico em que a resposta cai abaixo de metade do pico), em p.p. por 10 p.p. de fração.
linhas = []
chaves = ["especificacao", "dependente", "motor", "tipo_choque"]
for valores, grupo in nao_linear[nao_linear["tipo_choque"].isin(TIPOS)].groupby(chaves):
    escalado = grupo.assign(coef=grupo["coef_esc"], li_90=grupo["li_90_esc"],
                            ls_90=grupo["ls_90_esc"])  # fmt: skip
    resumo = lp.resumo_irf(escalado.drop(columns="especificacao")).iloc[0].to_dict()
    linhas.append(dict(zip(chaves, valores, strict=True))
                  | {k: resumo[k] for k in ["h6", "h12", "h18", "horizontes_signif_90"]}
                  | {"horizontes_acima": int((grupo["li_90"] > 0).sum()),
                     "horizontes_abaixo": int((grupo["ls_90"] < 0).sum())}
                  | lp.dissipacao(escalado, fracao=st["fracao_dissipacao"]))  # fmt: skip
resumo_nl = pd.DataFrame(linhas)
resumo_nl.to_csv(TABELAS / "lp_nao_linear_resumo.csv", index=False)
print(resumo_nl[resumo_nl["especificacao"] == "principal"].round(2).to_string(index=False))

# %%
NOMES_MOTOR = {"cinturao": "cinturão verde", "milho": "milho", "banana": "banana",
               "feijao": "feijão"}  # fmt: skip
modelos_ordem = [f"{m['dependente']}|{m['motor']}" for m in nl["modelos"]]
rotulos_modelos = {
    f"{m['dependente']}|{m['motor']}": f"{visual.NOMES_GRUPOS[m['dependente']]} × "
    f"{NOMES_MOTOR[m['motor']]} (h = {m['h_teste']})"
    for m in nl["modelos"]
}  # fmt: skip
floresta = no_teste[no_teste["especificacao"] == "principal"].rename(
    columns={"coef_esc": "coef", "li_90_esc": "li_90", "ls_90_esc": "ls_90", "coef": "coef_bruto",
             "li_90": "li_90_bruto", "ls_90": "ls_90_bruto", "choque": "coluna"}
).rename(columns={"tipo_choque": "choque"})  # fmt: skip
fig = visual.floresta_choques(
    floresta, modelos_ordem, rotulos_modelos,
    {"seca": "seca (SPI < −1,5)", "excesso": "excesso de chuva (SPI > 1,5)",
     "calor": "calor (Tmax > 1,5 desvio)"},
    "Seca, excesso de chuva e calor: resposta acumulada no horizonte da hipótese",
    "Resposta acumulada do preço (p.p.) a +10 p.p. da produção (ou das células do cinturão "
    "verde) na condição extrema, com os três choques no mesmo modelo; traço = banda HAC de 90%. "
    "Horizonte de cada linha fixado antes de estimar. Frações concentradas em 2012–2023 (R23).",
    "variação acumulada do preço (p.p.) por +10 p.p. de fração em condição extrema",
)  # fmt: skip
fig.savefig(FIGURAS / "lp_nao_linear_assimetria.png")


# %% [markdown]
# ## Parte 6b — Por área do IPCA: cinturão verde × regiões de monocultura
#
# Especificação registrada antes de estimar (`modelos.local_projections.areas`). Em cada área,
# os três choques (−SPI, padronizados pelo desvio-padrão amostral) entram juntos: cinturão
# verde da área (SPI-3, 150 km), CWI do milho no Centro-Oeste e no Matopiba (SPI-6). Preços:
# índice do IPCA da própria área; controles macro nacionais.

# %%
ar = cfg["areas"]
AREAS, DEPS = ar["areas"], ar["dependentes"]
ESC = ar["choques"]
CHOQUES_AREA = ["seca_local", "seca_centro_oeste", "seca_matopiba"]
precos = ipca[ipca["area_id"].isin(AREAS) & ipca["categoria"].isin(DEPS)]
precos = precos[(precos["mes"] >= inicio_janela) & (precos["mes"] <= FIM)]
log_area = precos.assign(valor=np.log(precos["indice"])).pivot_table(
    index=["area_id", "mes"], columns="categoria", values="valor"
)
log_area.columns = [f"log_{c}" for c in log_area.columns]
base_areas = log_area.reset_index()

cv = lakehouse.ler("cinturao_verde_mensal", config)
local = cv[(cv["raio_km"] == ESC["local"]["raio_km"]) & cv["area_id"].isin(AREAS)]
base_areas = base_areas.merge(
    local[["area_id", "mes", ESC["local"]["spi"]]].rename(columns={ESC["local"]["spi"]: "local"}),
    on=["area_id", "mes"], how="left",
)  # fmt: skip
cwi_regional = lakehouse.ler("cwi_mensal", config)
cwi_regional = cwi_regional[cwi_regional["versao"] == "defasado"]
for nome in ["centro_oeste", "matopiba"]:
    c = ESC[nome]
    filtro = (cwi_regional["regiao"] == c["regiao"]) & (cwi_regional["cultura"] == c["cultura"])
    serie = cwi_regional[filtro].set_index("mes")[c["spi"]].rename(nome)
    base_areas = base_areas.merge(serie, left_on="mes", right_index=True, how="left")
base_areas = base_areas.merge(
    base[cfg["controles_macro"]], left_on="mes", right_index=True, how="left"
)
# Fração das células do cinturão de cada área em seca, excesso e calor (SPI-3, como na 6a).
fracao_area = cinturao_verde.agregar(indicadores["spi_3"], selecao, TIPOS)
base_areas = base_areas.merge(
    fracao_area[["area_id", "mes", *TIPOS]].rename(columns={t: f"fracao_{t}_local" for t in TIPOS}),
    on=["area_id", "mes"], how="left",
)  # fmt: skip

# Padronização: −SPI / desvio-padrão na janela (por área, no caso do cinturão).
na_janela_area = (base_areas["mes"] >= INICIO) & (base_areas["mes"] <= FIM)
dp_choques_area = {}
for bruto in ["local", "centro_oeste", "matopiba"]:
    dp = base_areas[na_janela_area].groupby("area_id")[bruto].std()
    dp_choques_area[bruto] = dp
    base_areas[f"seca_{bruto}"] = (
        cfg["sinal_choque"] * base_areas[bruto] / base_areas["area_id"].map(dp)
    )
base_areas = base_areas.sort_values(["area_id", "mes"]).reset_index(drop=True)
lakehouse.salvar(base_areas, "base_analise_areas", config)
print(base_areas.groupby("area_id")["mes"].agg(["min", "max", "count"]))
print("faltantes:", base_areas.isna().sum()[lambda s: s > 0].to_dict())

# Lado do clima, por área (sem preços): dispersão, correlações e frações (R2, R4, R5).
R2_FRACO = {"belem", "salvador", "rio_de_janeiro"}  # áreas da R2 entre as 11
janela = base_areas[na_janela_area]
celulas_area = selecao.groupby("area_id")["celula_id"].nunique()
clima_areas = pd.DataFrame(
    {"dp_spi_local": dp_choques_area["local"],
     "corr_local_centro_oeste": janela.groupby("area_id")[["local", "centro_oeste"]].corr()
     .xs("local", level=1)["centro_oeste"],
     "corr_local_matopiba": janela.groupby("area_id")[["local", "matopiba"]].corr()
     .xs("local", level=1)["matopiba"],
     "celulas_150km": celulas_area,
     **{f"media_fracao_{t}": janela.groupby("area_id")[f"fracao_{t}_local"].mean() for t in TIPOS},
     "r2_nasa_fraco": [a in R2_FRACO for a in dp_choques_area["local"].index]}
).reindex(AREAS)  # fmt: skip
clima_areas.to_csv(TABELAS / "choques_areas_resumo.csv")
print(clima_areas.round(2).to_string())
print("DP Centro-Oeste:", round(dp_choques_area["centro_oeste"].iloc[0], 3),
      "| DP Matopiba:", round(dp_choques_area["matopiba"].iloc[0], 3))  # fmt: skip

# %% [markdown]
# ### Local Projections por área
#
# 11 áreas × 4 grupos × 2 especificações, cada uma com os três choques e os contrastes
# local − Centro-Oeste e local − Matopiba (mesma matriz HAC). Coeficientes por 1
# desvio-padrão de cada choque.

# %%
contrastes_area = [("seca_local", "seca_centro_oeste"), ("seca_local", "seca_matopiba")]
partes_area = []
for area in AREAS:
    dados_area = base_areas[base_areas["area_id"] == area].set_index("mes")
    for dep in DEPS:
        for nome in ar["especificacoes"]:
            espec = especificacao(nome, CHOQUES_AREA, dep, contrastes=contrastes_area)
            irf = lp.estimar(dados_area, espec, INICIO, FIM)
            partes_area.append(irf.assign(area_id=area, dependente=dep, especificacao=nome))
lp_areas = pd.concat(partes_area, ignore_index=True)
lp_areas["h_teste"] = lp_areas["dependente"].map(ar["h_teste"])
lp_areas.to_csv(TABELAS / "lp_areas.csv", index=False)
lakehouse.salvar(lp_areas, "lp_areas", config)
print(f"{lp_areas.groupby(['area_id', 'dependente', 'especificacao']).ngroups} modelos; "
      f"n de {lp_areas['n'].max()} a {lp_areas['n'].min()}")  # fmt: skip

# %% [markdown]
# ### Teste no horizonte registrado e classificação
#
# Classificação (principal, banda de 90% em h_teste, qualquer sinal): cinturão verde, monocultura,
# ambos ou nenhum. Por choque e grupo, nº de áreas com banda fora de zero contra o esperado por
# acaso (binomial com 11 áreas e 10%; as áreas não são independentes, então o p-valor é só uma
# referência).

# %%
from scipy import stats  # noqa: E402

no_h = lp_areas[lp_areas["horizonte"] == lp_areas["h_teste"]].copy()
no_h["fora_de_zero"] = (no_h["li_90"] > 0) | (no_h["ls_90"] < 0)
no_h.to_csv(TABELAS / "lp_areas_teste.csv", index=False)


def classificar(linhas: pd.DataFrame) -> str:
    sig = linhas.set_index("choque")["fora_de_zero"]
    local, mono = sig["seca_local"], sig["seca_centro_oeste"] or sig["seca_matopiba"]
    return {(True, False): "cinturão verde", (False, True): "monocultura",
            (True, True): "ambos", (False, False): "nenhum"}[(local, mono)]  # fmt: skip


linhas = []
for (area, dep, nome), grupo in no_h.groupby(["area_id", "dependente", "especificacao"]):
    coefs = grupo.set_index("choque")
    linha = {"area_id": area, "dependente": dep, "especificacao": nome,
             "h_teste": ar["h_teste"][dep], "classe": classificar(grupo)}  # fmt: skip
    for choque, rotulo in [("seca_local", "local"), ("seca_centro_oeste", "co"),
                           ("seca_matopiba", "matopiba"),
                           ("seca_local − seca_centro_oeste", "local_menos_co"),
                           ("seca_local − seca_matopiba", "local_menos_matopiba")]:  # fmt: skip
        linha |= {f"{rotulo}_coef": coefs.loc[choque, "coef"],
                  f"{rotulo}_li_90": coefs.loc[choque, "li_90"],
                  f"{rotulo}_ls_90": coefs.loc[choque, "ls_90"],
                  f"{rotulo}_p": coefs.loc[choque, "p_valor"]}  # fmt: skip
    linhas.append(linha)
classificacao = pd.DataFrame(linhas)
classificacao["r2_nasa_fraco"] = classificacao["area_id"].isin(R2_FRACO)
classificacao.to_csv(TABELAS / "lp_areas_classificacao.csv", index=False)
tabela_classes = classificacao.pivot_table(
    index=["especificacao", "area_id"], columns="dependente", values="classe", aggfunc="first"
)
print(tabela_classes.reindex(AREAS, level=1).to_string())

linhas = []
for (dep, nome, choque), grupo in no_h.groupby(["dependente", "especificacao", "choque"]):
    acima, abaixo = int((grupo["li_90"] > 0).sum()), int((grupo["ls_90"] < 0).sum())
    linhas.append(
        {"dependente": dep, "especificacao": nome, "choque": choque, "h_teste": ar["h_teste"][dep],
         "areas": len(grupo), "acima": acima, "abaixo": abaixo, "fora_de_zero": acima + abaixo,
         "esperado_acaso": 0.10 * len(grupo),
         "p_binomial": stats.binom.sf(acima + abaixo - 1, len(grupo), 0.10)}
    )  # fmt: skip
contagem = pd.DataFrame(linhas)
contagem.to_csv(TABELAS / "lp_areas_contagem.csv", index=False)
print(contagem.round(3).to_string(index=False))

# Perfil completo (h = 0 a 18): horizontes com banda acima/abaixo de zero, por área e choque.
perfil = (
    lp_areas.assign(acima=lp_areas["li_90"] > 0, abaixo=lp_areas["ls_90"] < 0)
    .groupby(["especificacao", "dependente", "choque", "area_id"])[["acima", "abaixo"]].sum()
    .reset_index()
)  # fmt: skip
print(
    perfil[perfil["especificacao"] == "principal"]
    .pivot_table(index=["dependente", "area_id"], columns="choque", values=["acima", "abaixo"])
    .to_string()
)

# %% [markdown]
# ### Painéis das 11 áreas
#
# Com efeito fixo de mês (todo choque comum absorvido): resposta ao clima local, linear (−SPI-3
# padronizado) e não linear (fração das células do cinturão em seca, excesso e calor, por +10
# p.p.). Sem efeito fixo de mês: os três choques, como na comparação por área. Erros de
# Driscoll-Kraay.

# %%
partes_painel = []
for dep in DEPS:
    espec = especificacao("principal", ["seca_local"], dep)
    irf = lp.estimar_painel(base_areas, espec, INICIO, FIM, efeito_tempo=True)
    partes_painel.append(irf.assign(painel="efeito_tempo_linear", dependente=dep,
                                    especificacao="principal"))  # fmt: skip
    nomes_nl = [f"fracao_{t}_local" for t in TIPOS]
    contrastes_nl = [(f"fracao_{a}_local", f"fracao_{b}_local") for a, b in nl["contrastes"]]
    espec = especificacao("principal", nomes_nl, dep, contrastes=contrastes_nl)
    irf = lp.estimar_painel(base_areas, espec, INICIO, FIM, efeito_tempo=True)
    partes_painel.append(irf.assign(painel="efeito_tempo_nao_linear", dependente=dep,
                                    especificacao="principal"))  # fmt: skip
    for nome in ar["especificacoes"]:
        espec = especificacao(nome, CHOQUES_AREA, dep, contrastes=contrastes_area)
        irf = lp.estimar_painel(base_areas, espec, INICIO, FIM)
        partes_painel.append(irf.assign(painel="sem_efeito_tempo", dependente=dep,
                                        especificacao=nome))  # fmt: skip
painel = pd.concat(partes_painel, ignore_index=True)
painel["h_teste"] = painel["dependente"].map(ar["h_teste"])
# Frações: resposta por +10 p.p. (como na 6a); choques padronizados: já por 1 desvio-padrão.
fracao = painel["choque"].str.startswith("fracao_")
for coluna in ["coef", "li_68", "ls_68", "li_90", "ls_90"]:
    painel[f"{coluna}_esc"] = painel[coluna].where(~fracao, painel[coluna] * nl["escala_fracao"])
painel.to_csv(TABELAS / "lp_painel.csv", index=False)
lakehouse.salvar(painel, "lp_painel", config)

linhas = []
for (pn, dep, nome, choque), grupo in painel.groupby(["painel", "dependente", "especificacao",
                                                        "choque"]):  # fmt: skip
    teste = grupo.set_index("horizonte").loc[ar["h_teste"][dep]]
    escalado = grupo.assign(coef=grupo["coef_esc"], li_90=grupo["li_90_esc"],
                            ls_90=grupo["ls_90_esc"])  # fmt: skip
    linhas.append(
        {"painel": pn, "dependente": dep, "especificacao": nome, "choque": choque,
         "h_teste": ar["h_teste"][dep], "coef_h_teste": teste["coef_esc"],
         "li_90_h_teste": teste["li_90_esc"], "ls_90_h_teste": teste["ls_90_esc"],
         "p_h_teste": teste["p_valor"],
         "horizontes_acima": int((grupo["li_90"] > 0).sum()),
         "horizontes_abaixo": int((grupo["ls_90"] < 0).sum()),
         **{k: v for k, v in lp.dissipacao(escalado).items() if k != "persistencia_h18"},
         "n_h0": int(grupo["n"].max())}
    )  # fmt: skip
resumo_painel = pd.DataFrame(linhas)
resumo_painel.to_csv(TABELAS / "lp_painel_resumo.csv", index=False)
print(resumo_painel.round(3).to_string(index=False))

# %% [markdown]
# ### Figuras da 6b
#
# Floresta por área no horizonte registrado (tabela: `lp_areas_teste.csv`) e respostas dos
# painéis (tabela: `lp_painel.csv`). Sinal: choque = −SPI; coeficiente negativo = o preço sobe
# com chuva acima do normal.

# %%
NOMES_AREAS = {a["id"]: a["nome"] for a in config["ipca"]["areas"]["lista"]}
rotulos_areas = {
    a: NOMES_AREAS[a] + (" (R2)" if a in R2_FRACO else "")
    + (" (cinturão no CO)" if a in {"goiania", "brasilia"} else "")
    for a in AREAS
}  # fmt: skip
nomes_tres = {"seca_local": "cinturão verde da área (SPI-3, 150 km)",
              "seca_centro_oeste": "milho no Centro-Oeste (CWI SPI-6)",
              "seca_matopiba": "milho no Matopiba (CWI SPI-6)"}  # fmt: skip
floresta_areas = no_h[(no_h["especificacao"] == "principal") & no_h["choque"].isin(nomes_tres)]
floresta_areas = floresta_areas.assign(painel=floresta_areas["dependente"],
                                       modelo=floresta_areas["area_id"])  # fmt: skip
fig = visual.floresta_grade(
    floresta_areas, DEPS,
    {d: f"{visual.NOMES_GRUPOS[d]} (h = {ar['h_teste'][d]})" for d in DEPS},
    AREAS, rotulos_areas, nomes_tres,
    "Cada área do IPCA: clima do cinturão verde × regiões de monocultura",
    "Resposta acumulada do preço de cada área (p.p.) a 1 desvio-padrão de seca (−SPI) em cada "
    "região, com os três choques no mesmo modelo, no horizonte registrado antes de estimar; "
    "traço = banda HAC de 90%. Valor negativo: o preço sobe com chuva acima do normal (R17). "
    "Amostra: 2000-09 a 2023-06. R2: NASA POWER fraco na área; Goiânia e Brasília: cinturão "
    "dentro do Centro-Oeste.",
    "variação acumulada do preço (p.p.) por 1 desvio-padrão de seca",
)  # fmt: skip
fig.savefig(FIGURAS / "lp_areas_floresta.png")

NOMES_LOCAIS = {"seca_local": "seca local (−SPI-3, 1 desvio-padrão)",
                "fracao_seca_local": "+10 p.p. do cinturão em seca",
                "fracao_excesso_local": "+10 p.p. do cinturão em excesso de chuva",
                "fracao_calor_local": "+10 p.p. do cinturão em calor"}  # fmt: skip
efeito_tempo = painel[painel["painel"].str.startswith("efeito_tempo")]
paineis_locais = [(d, c) for d in DEPS for c in NOMES_LOCAIS]
fig = visual.irf_grade(
    efeito_tempo, paineis_locais,
    {(d, c): f"{visual.NOMES_GRUPOS[d]}\n{NOMES_LOCAIS[c]}" for d, c in paineis_locais},
    "Painel das 11 áreas com efeito fixo de mês: só o clima próprio de cada cinturão",
    "Resposta acumulada do preço de cada área (p.p.). Efeito fixo de cada mês absorve todo "
    "choque comum às áreas (CWI, ENSO, câmbio, commodities): a estimativa vem só da diferença de "
    "clima entre cinturões no mesmo mês. 1ª coluna: −SPI-3 padronizado; demais: frações (as três "
    "no mesmo modelo). Bandas de Driscoll-Kraay; n = 3.014 em h = 0.",
    sufixo="_esc", janelas=hipoteses,
)  # fmt: skip
fig.savefig(FIGURAS / "lp_painel_local.png")

sem_tempo = painel[painel["painel"] == "sem_efeito_tempo"]
paineis_tres = [(d, c) for d in DEPS for c in CHOQUES_AREA]
fig = visual.irf_grade(
    sem_tempo[sem_tempo["especificacao"] == "principal"], paineis_tres,
    {(d, c): f"{visual.NOMES_GRUPOS[d]}\nseca: {nomes_tres[c]}" for d, c in paineis_tres},
    "Painel das 11 áreas: cinturão verde × Centro-Oeste × Matopiba no mesmo modelo",
    "Resposta acumulada do preço de cada área (p.p.) a 1 desvio-padrão de seca (−SPI) em cada "
    "região. Linha tracejada: com câmbio, IC-Br e diesel (R21). Efeito fixo de área por mês do "
    "ano e tendência por área; bandas de Driscoll-Kraay. Amostra: 2000-09 a 2023-06.",
    sufixo="_esc", ncols=3, janelas=hipoteses,
    comparacao=sem_tempo[sem_tempo["especificacao"] == "com_controles"],
    nome_comparacao="com controles",
)  # fmt: skip
fig.savefig(FIGURAS / "lp_painel_monocultura.png")


# %% [markdown]
# ## Parte 6c — ENSO × anomalias locais (pergunta 4)
#
# Especificação registrada antes de estimar (`modelos.local_projections.enso` e
# `modelos.previsao`). Índices com a defasagem de disponibilidade (R12): MEI v2 de t e ONI de
# t − 1; positivo = El Niño. Nas Local Projections, padronizados pelo desvio-padrão na janela
# (resposta por 1 desvio-padrão); na previsão, em nível (a previsão não depende da escala).

# %%
en = cfg["enso"]
pr = config["modelos"]["previsao"]
INDICES = list(en["indices"])
NOMES_INDICES = {"mei": "MEI v2 (t)", "oni": "ONI (t − 1)"}
dp_enso = {}
for nome, indice in en["indices"].items():
    bruto = base[indice["coluna"]].shift(indice["defasagem"])
    dp_enso[nome] = bruto[na_amostra].std()
    base[f"enso_{nome}_bruto"] = bruto
    base[f"enso_{nome}"] = bruto / dp_enso[nome]
print("Desvio-padrão na janela:", {k: round(v, 3) for k, v in dp_enso.items()}, "| correlação:",
      round(base.loc[na_amostra, ["enso_mei", "enso_oni"]].corr().iloc[0, 1], 3))  # fmt: skip

# %% [markdown]
# ### Escala dos choques: desvio-padrão do nível × inovação típica (R29)
#
# O coeficiente da LP mantém fixas as 12 defasagens do choque: é a resposta à parte do choque
# em t que o passado não prevê (a inovação). Em choques persistentes, 1 desvio-padrão do nível
# (escala das Partes 5 a 6b, R22) equivale a várias inovações típicas. `fator_conversao`
# multiplica as respostas publicadas para obtê-las por 1 inovação típica. Tabela acrescentada
# depois de ver a diferença de escala entre ONI e MEI nas respostas ao ENSO; não muda
# nenhuma estimativa, sinal ou significância.

# %%
LAGS = cfg["lags_choque"]


def escala_choque(parte: str, choque: str, serie: pd.Series, dp_nivel: float,
                  escala_publicada: str, unidade: float) -> dict:  # fmt: skip
    """Inovação típica do choque e fator que converte a resposta publicada (por `unidade`)."""
    dp_inovacao = lp.desvio_inovacao(serie, LAGS, INICIO, FIM)
    janela_serie = serie[(serie.index >= INICIO) & (serie.index <= FIM)]
    return {"parte": parte, "choque": choque, "dp_nivel": dp_nivel, "dp_inovacao": dp_inovacao,
            "razao_inovacao_nivel": dp_inovacao / dp_nivel, "ar1": janela_serie.autocorr(),
            "escala_publicada": escala_publicada,
            "fator_conversao": dp_inovacao / unidade}  # fmt: skip


linhas = [escala_choque("5 e 6a", c, base[c], dp, "1 desvio-padrão do nível", dp)
          for c, dp in desvios.items()]  # fmt: skip
linhas += [escala_choque("6a", c, base[c], dp, "+10 p.p. de fração", nl["escala_fracao"])
           for c, dp in desvio_fracao.items()]  # fmt: skip
for area in AREAS:
    dados_area = base_areas[base_areas["area_id"] == area].set_index("mes")
    linhas.append(escala_choque("6b", f"seca_local|{area}", dados_area["seca_local"], 1.0,
                                "1 desvio-padrão do nível", 1.0))  # fmt: skip
# CO e Matopiba: a mesma série em todas as áreas.
for regional in ["seca_centro_oeste", "seca_matopiba"]:
    linhas.append(escala_choque("6b", regional, dados_area[regional], 1.0,
                                "1 desvio-padrão do nível", 1.0))  # fmt: skip
linhas += [escala_choque("6c", f"enso_{i}", base[f"enso_{i}"], 1.0, "1 desvio-padrão do nível",
                         1.0) for i in INDICES]  # fmt: skip
escala_choques = pd.DataFrame(linhas)
escala_choques.to_csv(TABELAS / "escala_choques.csv", index=False)
print(escala_choques.round(3).to_string(index=False))
fator_enso = escala_choques.set_index("choque")["fator_conversao"]

# %% [markdown]
# ### Lado do clima: quanto o ENSO explica do clima local
#
# R² da regressão de cada indicador local (janela 2000-09 a 2023-06) no índice ENSO em t e em
# 12 defasagens. Descritivo: mede o quanto o ENSO antecipa o clima local que entra nos modelos.

# %%
ALVOS_CLIMA = {
    "cwi_milho_spi_6": ("nacional", "milho (CWI SPI-6)"),
    "cwi_milho_2a_safra_spi_6": ("nacional", "milho 2ª safra (CWI SPI-6)"),
    "cwi_soja_spi_6": ("nacional", "soja (CWI SPI-6)"),
    "cwi_banana_spi_3": ("nacional", "banana (CWI SPI-3)"),
    "cwi_feijao_spi_3": ("nacional", "feijão (CWI SPI-3)"),
    "cinturao_spi_3": ("nacional", "cinturão verde médio (SPI-3)"),
    "cwi_milho_spi_6_centro_oeste": ("regional", "milho no Centro-Oeste (CWI SPI-6)"),
    "cwi_milho_spi_6_matopiba": ("regional", "milho no Matopiba (CWI SPI-6)"),
}
linhas = []
for indice in INDICES:
    x = base[f"enso_{indice}_bruto"]
    for alvo_clima, (escopo, rotulo) in ALVOS_CLIMA.items():
        ajuste = pv.r2_defasagens(base.loc[na_amostra, alvo_clima], x, cfg["lags_choque"])
        linhas.append({"indice": indice, "escopo": escopo, "alvo": alvo_clima, "rotulo": rotulo,
                       **ajuste})  # fmt: skip
    for area in AREAS:
        serie = janela[janela["area_id"] == area].set_index("mes")["local"]
        ajuste = pv.r2_defasagens(serie, x, cfg["lags_choque"])
        linhas.append({"indice": indice, "escopo": "cinturão da área", "alvo": area,
                       "rotulo": NOMES_AREAS[area], **ajuste})  # fmt: skip
enso_clima = pd.DataFrame(linhas)
enso_clima.to_csv(TABELAS / "enso_clima_local.csv", index=False)
print(enso_clima.pivot_table(index=["escopo", "rotulo"], columns="indice",
                             values=["r2_ajustado", "corr_t"]).round(3).to_string())  # fmt: skip

# %% [markdown]
# ### Resposta dos grupos ao ENSO
#
# Local Projections com o índice ENSO como choque (mesma especificação das Partes 5 e 6a), por
# 1 desvio-padrão, principal e com controles. Sem hipótese de sinal.

# %%
partes_enso = []
for dep in en["dependentes"]:
    for indice in INDICES:
        for nome in en["especificacoes"]:
            irf = lp.estimar(base, especificacao(nome, [f"enso_{indice}"], dep), INICIO, FIM)
            partes_enso.append(irf.assign(dependente=dep, indice=indice, especificacao=nome))
lp_enso = pd.concat(partes_enso, ignore_index=True)
# Por 1 desvio-padrão do nível (choque padronizado) e por 1 inovação típica (R29).
lp_enso["fator_inovacao"] = lp_enso["choque"].map(fator_enso)
for coluna in ["coef", "li_68", "ls_68", "li_90", "ls_90"]:
    lp_enso[f"{coluna}_inov"] = lp_enso[coluna] * lp_enso["fator_inovacao"]
lp_enso.to_csv(TABELAS / "lp_enso.csv", index=False)
lakehouse.salvar(lp_enso, "lp_enso", config)

linhas = []
for (dep, indice, nome), grupo in lp_enso.groupby(["dependente", "indice", "especificacao"]):
    por_h = grupo.set_index("horizonte")
    pico = por_h.loc[por_h["coef"].abs().idxmax()]
    linhas.append(
        {"dependente": dep, "indice": indice, "especificacao": nome,
         "h6": por_h.loc[6, "coef"], "h12": por_h.loc[12, "coef"],
         "li_90_h12": por_h.loc[12, "li_90"], "ls_90_h12": por_h.loc[12, "ls_90"],
         "h18": por_h.loc[18, "coef"], "pico_coef": pico["coef"], "pico_h": int(pico.name),
         "fator_inovacao": pico["fator_inovacao"], "h6_inov": por_h.loc[6, "coef_inov"],
         "h12_inov": por_h.loc[12, "coef_inov"], "pico_coef_inov": pico["coef_inov"],
         "horizontes_acima": int((grupo["li_90"] > 0).sum()),
         "horizontes_abaixo": int((grupo["ls_90"] < 0).sum())}
    )  # fmt: skip
resumo_enso = pd.DataFrame(linhas)
resumo_enso.to_csv(TABELAS / "lp_enso_resumo.csv", index=False)
print(resumo_enso.round(2).to_string(index=False))

# %% [markdown]
# ### O canal da ração sobrevive ao controle por ENSO? (R21)
#
# Mesmo modelo da 6a (principal), com o índice ENSO (valor em t e 12 defasagens) no mesmo
# modelo. Leitura fixada antes: "sobrevive" se a banda de 90% do milho em h = 12 continuar
# acima de zero. Respostas do milho por 1 desvio-padrão (R22).

# %%
cm = en["controle_milho"]
partes_cm = []
for dep, motor in cm["pares"]:
    milho = f"seca_{motor}"
    for controle in ["nenhum", *cm["indices"]]:
        choques_modelo = [milho] + ([f"enso_{controle}"] if controle != "nenhum" else [])
        irf = lp.estimar(base, especificacao("principal", choques_modelo, dep), INICIO, FIM)
        partes_cm.append(irf.assign(dependente=dep, motor=motor, controle_enso=controle))
controle_milho = pd.concat(partes_cm, ignore_index=True)
e_milho = controle_milho["choque"].str.startswith("seca_")
controle_milho["dp_choque"] = controle_milho["choque"].map(desvios).where(e_milho, 1.0)
for coluna in ["coef", "li_68", "ls_68", "li_90", "ls_90"]:
    controle_milho[f"{coluna}_dp"] = controle_milho[coluna] * controle_milho["dp_choque"]
controle_milho.to_csv(TABELAS / "lp_enso_controle_milho.csv", index=False)

# Conferência: sem controle, o modelo é o mesmo da Parte 5 (alimentação) e da 6a (aves, carnes).
referencia = pd.concat([
    irfs[irfs["especificacao"] == "principal"].assign(dependente=dependente),
    setorial[(setorial["tipo"] == "acumulada") & (setorial["especificacao"] == "principal")],
])  # fmt: skip
sem_controle = controle_milho[controle_milho["controle_enso"] == "nenhum"]
conferencia = sem_controle.merge(referencia, on=["dependente", "choque", "horizonte"],
                                 suffixes=("", "_ref"))  # fmt: skip
assert len(conferencia) == len(sem_controle)
assert np.allclose(conferencia["coef"], conferencia["coef_ref"], atol=1e-10)

h_cm = cm["h_teste"]
linhas = []
for (dep, motor), grupo in controle_milho[e_milho].groupby(["dependente", "motor"]):
    base_h = grupo[(grupo["controle_enso"] == "nenhum") & (grupo["horizonte"] == h_cm)]
    coef_sem = base_h["coef_dp"].iloc[0]
    for controle, g in grupo.groupby("controle_enso"):
        no_h_cm = g.set_index("horizonte").loc[h_cm]
        enso_h = controle_milho[(controle_milho["dependente"] == dep)
                                & (controle_milho["motor"] == motor)
                                & (controle_milho["controle_enso"] == controle)
                                & (controle_milho["choque"] == f"enso_{controle}")
                                & (controle_milho["horizonte"] == h_cm)]  # fmt: skip
        linhas.append(
            {"dependente": dep, "motor": motor, "controle_enso": controle, "h_teste": h_cm,
             "coef_dp": no_h_cm["coef_dp"], "li_90_dp": no_h_cm["li_90_dp"],
             "ls_90_dp": no_h_cm["ls_90_dp"], "p_valor": no_h_cm["p_valor"],
             "variacao_pct": 100 * (no_h_cm["coef_dp"] / coef_sem - 1),
             "sobrevive": bool(no_h_cm["li_90_dp"] > 0),
             "horizontes_acima": int((g["li_90"] > 0).sum()),
             "enso_coef_h_teste": enso_h["coef"].iloc[0] if len(enso_h) else np.nan,
             "enso_li_90_h_teste": enso_h["li_90"].iloc[0] if len(enso_h) else np.nan,
             "enso_ls_90_h_teste": enso_h["ls_90"].iloc[0] if len(enso_h) else np.nan}
        )  # fmt: skip
resumo_cm = pd.DataFrame(linhas)
resumo_cm.to_csv(TABELAS / "lp_enso_controle_milho_resumo.csv", index=False)
print(resumo_cm.round(3).to_string(index=False))

# %% [markdown]
# ### Previsão pseudo fora da amostra
#
# Regressão direta por horizonte, janela expansiva (estimação de 2000-09 até a origem − h),
# origens de 2011-12 a 2023-06 − h. Benchmark: constante, dummies de mês e inflação de t a
# t − 11; cada modelo acrescenta preditores em t, t − 1 e t − 2. Clima local = motor principal
# do grupo (alimentação no domicílio: cinturão + milho).

# %%
PRIMEIRA_ORIGEM = pd.Timestamp(pr["primeira_origem"])
NIVEL = pr["nivel_teste"]
enso_prev = {indice: [f"enso_{indice}_bruto"] for indice in INDICES}
partes_prev = []
for dep in en["dependentes"]:
    local = pr["local"][dep]
    preditores = {"benchmark": [], "oni": enso_prev["oni"], "mei": enso_prev["mei"],
                  "local": local, "local_oni": local + enso_prev["oni"],
                  "local_mei": local + enso_prev["mei"]}  # fmt: skip
    for modelo in pr["modelos"]:
        espec = pv.EspecificacaoPrevisao(
            log_preco=f"log_{dep}", preditores=preditores[modelo],
            lags_inflacao=pr["lags_inflacao"], lags_preditor=pr["lags_preditor"],
            dummies_mes=pr["dummies_mes"],
        )  # fmt: skip
        for h in pr["horizontes"]:
            prev = pv.prever(base, espec, h, INICIO, PRIMEIRA_ORIGEM, FIM)
            partes_prev.append(prev.assign(dependente=dep, modelo=modelo))
previsoes = pd.concat(partes_prev, ignore_index=True)
previsoes["mes_alvo"] = (
    previsoes["origem"].dt.to_period("M") + previsoes["horizonte"].to_numpy()
).dt.to_timestamp()
previsoes.to_csv(TABELAS / "enso_previsoes.csv.gz", index=False,
                 compression={"method": "gzip", "mtime": 0})  # fmt: skip
print(f"{len(previsoes)} previsões; origens por horizonte:",
      previsoes.groupby("horizonte")["origem"].nunique().to_dict(),
      "| alvos de", previsoes["mes_alvo"].min().strftime("%Y-%m"), "a",
      previsoes["mes_alvo"].max().strftime("%Y-%m"),
      "| n de estimação de", previsoes["n_estimacao"].min(), "a",
      previsoes["n_estimacao"].max())  # fmt: skip

# %% [markdown]
# ### Avaliação: Clark-West (aninhados) e Diebold-Mariano (ENSO × local)
#
# R² fora da amostra = 1 − EQM do modelo / EQM do benchmark (em %). Clark-West unilateral:
# oni, mei e local contra o benchmark; local_oni e local_mei contra o local. Diebold-Mariano
# (correção HLN, bilateral): oni × local e mei × local. Nível 10%; 32 grupos × horizontes por
# comparação, então ~3 rejeições por acaso em cada uma.

# %%
COMPARACOES = {"oni": "benchmark", "mei": "benchmark", "local": "benchmark",
               "local_oni": "local", "local_mei": "local"}  # fmt: skip
linhas, linhas_dm = [], []
for (dep, h), grupo in previsoes.groupby(["dependente", "horizonte"]):
    por_modelo = {m: g.set_index("origem").sort_index() for m, g in grupo.groupby("modelo")}
    bench = por_modelo["benchmark"]
    assert all(g.index.equals(bench.index) for g in por_modelo.values())
    linhas.append({"dependente": dep, "horizonte": h, "modelo": "benchmark",
                   "eqm": np.mean(bench["erro"] ** 2), "r2_oos_pct": 0.0,
                   "n": len(bench)})  # fmt: skip
    for modelo, restrito in COMPARACOES.items():
        amplo, menor = por_modelo[modelo], por_modelo[restrito]
        cw = pv.clark_west(amplo["alvo"], menor["previsao"], amplo["previsao"], h)
        linhas.append(
            {"dependente": dep, "horizonte": h, "modelo": modelo, "eqm": cw["eqm_amplo"],
             "r2_oos_pct": 100 * pv.r2_fora_amostra(amplo["erro"], bench["erro"]),
             "comparado_com": restrito,
             "r2_oos_comparado_pct": 100 * pv.r2_fora_amostra(amplo["erro"], menor["erro"]),
             "cw_estatistica": cw["estatistica"], "cw_p": cw["p_valor"], "n": cw["n"]}
        )  # fmt: skip
    for indice in INDICES:
        dm = pv.diebold_mariano(por_modelo[indice]["erro"], por_modelo["local"]["erro"], h)
        if dm["p_valor"] < NIVEL:
            classe = "ENSO superior" if dm["estatistica"] < 0 else "local superior"
        else:
            classe = "sem diferença"
        linhas_dm.append(
            {"dependente": dep, "horizonte": h, "indice": indice, "eqm_enso": dm["eqm_a"],
             "eqm_local": dm["eqm_b"], "razao_eqm": dm["eqm_a"] / dm["eqm_b"],
             "dm_estatistica": dm["estatistica"], "dm_p": dm["p_valor"], "classe": classe,
             "n": dm["n"]}
        )  # fmt: skip
avaliacao_prev = pd.DataFrame(linhas)
avaliacao_prev["vence_10pct"] = avaliacao_prev["cw_p"] < NIVEL
avaliacao_prev.to_csv(TABELAS / "enso_previsao_avaliacao.csv", index=False)
dm_prev = pd.DataFrame(linhas_dm)
dm_prev.to_csv(TABELAS / "enso_previsao_dm.csv", index=False)
print(avaliacao_prev.pivot_table(index=["dependente", "horizonte"], columns="modelo",
                                 values="r2_oos_pct").round(1).to_string())  # fmt: skip
print(avaliacao_prev.pivot_table(index=["dependente", "horizonte"], columns="modelo",
                                 values="cw_p").round(3).to_string())  # fmt: skip
contagem_prev = (
    avaliacao_prev.dropna(subset=["cw_p"]).groupby("modelo")
    .agg(comparado_com=("comparado_com", "first"), casos=("cw_p", "size"),
         vence_10pct=("vence_10pct", "sum"), r2_oos_mediano_pct=("r2_oos_pct", "median"),
         r2_oos_positivo=("r2_oos_pct", lambda s: int((s > 0).sum())))
    .reset_index()
)  # fmt: skip
contagem_prev["esperado_acaso"] = NIVEL * contagem_prev["casos"]
contagem_prev.to_csv(TABELAS / "enso_previsao_contagem.csv", index=False)
print(contagem_prev.round(2).to_string(index=False))
print(dm_prev.round(3).to_string(index=False))
print(dm_prev.groupby(["indice", "classe"]).size().unstack(fill_value=0))

# %% [markdown]
# ### Concentração do ganho no tempo (verificação descritiva)
#
# Acrescentada depois de ver os resultados (não registrada antes), para todos os 96 casos
# grupo × horizonte × preditor, não só os significativos: ganho = soma de (erro do
# benchmark)² − (erro do modelo)² por ano do alvo; quanto dele vem do ano de maior ganho.
# A avaliação sem 2020–2022 é a robustez registrada para a Parte 8.

# %%
linhas = []
for (dep, h), grupo in previsoes.groupby(["dependente", "horizonte"]):
    bench = grupo[grupo["modelo"] == "benchmark"].set_index("origem").sort_index()
    for modelo in ["local", "mei", "oni"]:
        prev = grupo[grupo["modelo"] == modelo].set_index("origem").sort_index()
        ganho = (bench["erro"] ** 2 - prev["erro"] ** 2).groupby(prev["mes_alvo"].dt.year).sum()
        linhas.append(
            {"dependente": dep, "horizonte": h, "modelo": modelo, "ganho_total": ganho.sum(),
             "ano_maior_ganho": int(ganho.idxmax()), "ganho_maior_ano": ganho.max(),
             "ganho_sem_maior_ano": ganho.sum() - ganho.max(),
             "anos_com_ganho": int((ganho > 0).sum()), "anos": len(ganho)}
        )  # fmt: skip
concentracao = pd.DataFrame(linhas).merge(
    avaliacao_prev[["dependente", "horizonte", "modelo", "r2_oos_pct", "cw_p"]],
    on=["dependente", "horizonte", "modelo"],
)  # fmt: skip
concentracao.to_csv(TABELAS / "enso_previsao_concentracao.csv", index=False)
print(concentracao[concentracao["cw_p"] < NIVEL].round(2).to_string(index=False))

# %% [markdown]
# ### Figuras da 6c
#
# Respostas ao ENSO (tabela: `lp_enso.csv`), canal da ração com e sem ENSO
# (`lp_enso_controle_milho_resumo.csv`), R² fora da amostra (`enso_previsao_avaliacao.csv`) e
# erro quadrático acumulado (`enso_erro_acumulado.csv`).

# %%
principal_enso = lp_enso[lp_enso["especificacao"] == "principal"]
paineis_enso = [(d, "enso_mei") for d in en["dependentes"]]
fig = visual.irf_grade(
    principal_enso[principal_enso["indice"] == "mei"], paineis_enso,
    {(d, "enso_mei"): visual.NOMES_GRUPOS[d] for d in en["dependentes"]},
    "Resposta de cada grupo a uma notícia de El Niño (MEI v2, 1 inovação típica)",
    "Resposta acumulada do preço (Brasil, p.p.) a 1 inovação típica do MEI v2 em t (a parte do "
    "índice que os 12 meses anteriores não preveem; positivo = rumo ao El Niño). Linha "
    "tracejada: ONI de t − 1 (média centrada, R12), na mesma escala. Amostra: t de 2000-09 a "
    "2023-06 − h; 12 defasagens do índice e da inflação, dummies de mês e tendência; bandas HAC "
    "de Newey-West. Por 1 desvio-padrão do nível: dividir por "
    f"{visual.numero(fator_enso['enso_mei'])} (MEI) ou {visual.numero(fator_enso['enso_oni'])} "
    "(ONI) (R29).",
    sufixo="_inov", comparacao=principal_enso[principal_enso["indice"] == "oni"].assign(
        choque="enso_mei"), nome_comparacao="ONI (t − 1)",
)  # fmt: skip
fig.savefig(FIGURAS / "lp_enso.png")

NOMES_PARES = {
    ("alimentacao_domicilio", "cwi_milho_spi_6"): "Alimentação no domicílio × milho",
    ("aves_ovos", "cwi_milho_spi_6"): "Aves e ovos × milho",
    ("aves_ovos", "cwi_milho_2a_safra_spi_6"): "Aves e ovos × milho 2ª safra",
    ("carnes", "cwi_milho_spi_6"): "Carnes × milho",
    ("carnes", "cwi_milho_2a_safra_spi_6"): "Carnes × milho 2ª safra",
}
floresta_cm = controle_milho[e_milho & controle_milho["horizonte"].isin([6, h_cm])]
floresta_cm = floresta_cm.assign(
    painel=floresta_cm["horizonte"].map(lambda h: f"h{h}"),
    modelo=floresta_cm["dependente"] + "|" + floresta_cm["motor"],
    choque=floresta_cm["controle_enso"],
).drop(columns=["coef", "li_90", "ls_90"]).rename(
    columns={"coef_dp": "coef", "li_90_dp": "li_90", "ls_90_dp": "ls_90"}
)  # fmt: skip
fig = visual.floresta_grade(
    floresta_cm, ["h6", f"h{h_cm}"], {"h6": "h = 6 meses", f"h{h_cm}": f"h = {h_cm} meses"},
    [f"{d}|{m}" for d, m in cm["pares"]], {f"{d}|{m}": n for (d, m), n in NOMES_PARES.items()},
    {"nenhum": "sem ENSO (6a)", "mei": "+ MEI v2", "oni": "+ ONI (t − 1)"},
    "Canal da ração com e sem controle por ENSO",
    "Resposta acumulada do preço (Brasil, p.p.) a 1 desvio-padrão de seca (−SPI-6) no CWI do "
    "milho, sem e com o índice ENSO (valor em t e 12 defasagens) no mesmo modelo; traço = banda "
    "HAC de 90%. Especificação principal da 6a. Leitura registrada: h = 12.",
    "p.p. por 1 desvio-padrão de seca no milho",
)  # fmt: skip
fig.savefig(FIGURAS / "lp_enso_controle_milho.png")

SERIES_PREV = {"local": "clima local", "mei": "MEI v2 (t)", "oni": "ONI (t − 1)"}
tabela_r2 = avaliacao_prev[avaliacao_prev["modelo"].isin(SERIES_PREV)].assign(
    painel=lambda d: d["dependente"], serie=lambda d: d["modelo"],
    valor=lambda d: d["r2_oos_pct"], destaque=lambda d: d["vence_10pct"],
)  # fmt: skip
NOMES_LOCAL = {
    d: " + ".join(NOMES_CHOQUES[f"seca_{c}"].split(" (")[0] for c in pr["local"][d])
    for d in en["dependentes"]
}  # fmt: skip
fig = visual.previsao_grade(
    tabela_r2, en["dependentes"],
    {d: f"{visual.NOMES_GRUPOS[d]}\nlocal: {NOMES_LOCAL[d]}" for d in en["dependentes"]},
    SERIES_PREV, "Previsão fora da amostra: ganho sobre o benchmark (2012–2023)",
    "R² fora da amostra = 1 − erro quadrático médio do modelo / o do benchmark (constante, "
    "dummies de mês e inflação dos 12 meses anteriores), em %. Positivo = o preditor melhora a "
    "previsão da inflação acumulada dos h meses seguintes. Regressão direta, janela expansiva "
    "desde 2000-09; alvos de 2012-01 a 2023-06. Marcador cheio: Clark-West unilateral p < 0,10. "
    "Dados finais, não em tempo real.",
    "R² fora da amostra (%)", "Clark-West p < 0,10",
)  # fmt: skip
fig.savefig(FIGURAS / "enso_previsao.png")

linhas = []
for dep, h in pr["erro_acumulado"]:
    caso = previsoes[(previsoes["dependente"] == dep) & (previsoes["horizonte"] == h)]
    bench = caso[caso["modelo"] == "benchmark"].set_index("origem").sort_index()
    for modelo in SERIES_PREV:
        prev = caso[caso["modelo"] == modelo].set_index("origem").sort_index()
        diferenca = (bench["erro"] ** 2 - prev["erro"] ** 2).cumsum()
        linhas.append(pd.DataFrame({"painel": f"{dep}|{h}", "dependente": dep, "horizonte": h,
                                    "serie": modelo, "mes": prev["mes_alvo"].to_numpy(),
                                    "valor": diferenca.to_numpy()}))  # fmt: skip
erro_acumulado = pd.concat(linhas, ignore_index=True)
erro_acumulado.to_csv(TABELAS / "enso_erro_acumulado.csv", index=False)
fig = visual.erro_acumulado_paineis(
    erro_acumulado, [f"{d}|{h}" for d, h in pr["erro_acumulado"]],
    {f"{d}|{h}": f"{visual.NOMES_GRUPOS[d]}, h = {h}" for d, h in pr["erro_acumulado"]},
    SERIES_PREV, "Onde o ganho de previsão aparece: erro acumulado contra o benchmark",
    "Soma acumulada, no mês do alvo, de (erro do benchmark)² − (erro do modelo)². Subida = o "
    "modelo erra menos que o benchmark naquele período; descida = erra mais. Casos fixados "
    "antes de rodar. Faixa cinza: 2020–2022 (pandemia e boom de commodities, R21).",
    "erro² acumulado: benchmark − modelo",
    faixa=(pd.Timestamp("2020-01-01"), pd.Timestamp("2022-12-01"), "2020–2022"),
)  # fmt: skip
fig.savefig(FIGURAS / "enso_erro_acumulado.png")
