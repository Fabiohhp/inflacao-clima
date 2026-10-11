# %% [markdown]
# # 07 — Testes de robustez
#
# **Parte 8a (Local Projections nacionais):** estabilidade dos coeficientes climáticos das
# Partes 5 e 6a sob os testes registrados antes de rodar (`modelos.robustez`): com controles,
# placebo com choque futuro, sem 2020–2022, janelas a partir de 2006-07 e 2012-01, SPI-3 ×
# SPI-6, pesos fixos de produção e limiares de ±1,0 e ±2,0 (versão não linear). Cada
# coeficiente-chave é lido no horizonte de teste do grupo e classificado como robusto, frágil,
# instável ou nulo (estável ou instável) pela regra do settings.
#
# Escalas como nas partes de origem: lineares por 1 desvio-padrão do choque na janela principal
# (R22, R29: os fatores por inovação típica estão em `escala_choques.csv`); não lineares por
# +10 p.p. de fração em condição extrema. Ressalvas centrais: R16, R21, R23, R26.
#
# **Parte 8b (áreas e soja):** robustez da 6b (cinturão de 300 km, painel com as 16 áreas,
# placebo nos painéis com efeito fixo de mês, inclusive a falsificação aves × calor local) e
# validação do CWI da soja com janela crítica por região (`modelos.robustez.areas_8b`).
#
# **Parte 8c (previsão e consolidação):** a previsão fora da amostra da 6c com janela móvel de
# 120 meses, avaliada sem os alvos que cruzam 2020–2022 e, em hortaliças e tubérculos, com o
# clima local como frações do cinturão (`modelos.robustez.previsao` e `previsao_8c`); por fim,
# as classificações da 8a, da 8b e da 8c numa tabela só.
#
# Rodar como script: `python notebooks/07_testes_robustez.py`.

# %%
import os

# BLAS com 1 thread (como no notebook 05): muitas regressões pequenas rodam mais rápido.
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
configurar_log(caminho_projeto("logs", config) / "07_robustez.log")
TABELAS = caminho_projeto("tabelas", config)
FIGURAS = caminho_projeto("figuras", config)
cfg = config["modelos"]["local_projections"]
rob = config["modelos"]["robustez"]
lpn = rob["lp_nacional"]
testes_cfg = lpn["testes"]
INICIO = pd.Timestamp(cfg["inicio_efetivo"])
FIM = pd.Timestamp(config["periodos"]["amostra_estimacao"]["fim"])
HORIZONTES = range(cfg["horizonte_min"], cfg["horizonte_max"] + 1)
EXCLUIR = (pd.Timestamp(rob["excluir_periodo"]["inicio"]),
           pd.Timestamp(rob["excluir_periodo"]["fim"]))  # fmt: skip
PLACEBO = testes_cfg["placebo"]
SINAL = cfg["sinal_choque"]
pd.set_option("display.width", 200, "display.max_columns", 30)

base = lakehouse.ler("base_analise_brasil", config).set_index("mes")
na_janela = (base.index >= INICIO) & (base.index <= FIM)

# CWI com pesos fixos de 2003-2007 (Parte 3), Brasil.
cwi_fixo = lakehouse.ler("cwi_mensal", config).query("versao == 'fixo' and regiao == 'brasil'")
for cultura, grupo in cwi_fixo.groupby("cultura"):
    for spi in ["spi_3", "spi_6"]:
        base[f"cwi_{cultura}_{spi}_fixo"] = grupo.set_index("mes")[spi]


def especificacao(dependente: str, choques: list[str], controles: bool = False,
                  horizontes: range = HORIZONTES) -> lp.EspecificacaoLP:  # fmt: skip
    """Especificação principal (ou com controles) das Partes 5 e 6a."""
    return lp.EspecificacaoLP(
        log_preco=f"log_{dependente}", choques=choques, horizontes=horizontes,
        lags_dependente=cfg["lags_dependente"], lags_choque=cfg["lags_choque"],
        lags_controles=cfg["lags_controles"], hac_lags_extra=cfg["hac_lags_extra"],
        niveis_confianca=(0.90,), controles=cfg["controles_macro"] if controles else [],
    )  # fmt: skip


def seca(coluna: str) -> str:
    """Cria (se preciso) a coluna de choque de seca = sinal × SPI e devolve o nome."""
    nome = f"seca_{coluna}"
    if nome not in base:
        base[nome] = SINAL * base[coluna]
    return nome


# %% [markdown]
# ## Lineares (Partes 5 e 6a): um choque por modelo

# %%
TESTES_LINEARES = ["principal", "com_controles", "sem_2020_2022", "janela_2006", "janela_2012",
                   "spi_alternativo", "pesos_fixos"]  # fmt: skip
JANELAS = {j["nome"]: pd.Timestamp(j["inicio"]) for j in testes_cfg["janelas"]}


def rodar_linear(dep: str, choque: str, teste: str) -> pd.DataFrame | None:
    """Resposta acumulada do teste, por 1 desvio-padrão (na janela principal) do choque usado."""
    coluna, inicio, extra, controles = choque, INICIO, {}, False
    if teste == "com_controles":
        controles = True
    elif teste == "sem_2020_2022":
        extra = {"excluir_periodo": EXCLUIR}
    elif teste in JANELAS:
        inicio = JANELAS[teste]
    elif teste == "spi_alternativo":
        coluna = testes_cfg["spi_alternativo"][choque]
    elif teste == "pesos_fixos":
        if not choque.startswith("cwi_"):
            return None  # o cinturão não tem pesos de produção
        coluna = f"{choque}_fixo"
    nome = seca(coluna)
    irf = lp.estimar(base, especificacao(dep, [nome], controles), inicio, FIM, **extra)
    dp = base.loc[na_janela, nome].std()
    for c in ["coef", "li_90", "ls_90"]:
        irf[c] = irf[c] * dp
    return irf.assign(dp_choque=dp, coluna=coluna)


irfs_lineares = []
for dep, choque, h in lpn["lineares"]:
    for teste in TESTES_LINEARES:
        irf = rodar_linear(dep, choque, teste)
        if irf is not None:
            irfs_lineares.append(irf.assign(familia="linear", dependente=dep, chave=choque,
                                            tipo="seca", h_teste=h, teste=teste))  # fmt: skip
irfs_lineares = pd.concat(irfs_lineares, ignore_index=True)

# Placebo: choque de t + 12 no lugar do de t; origens até 2022-06; h = 0 a 11.
H_PLACEBO = range(PLACEBO["horizontes"][0], PLACEBO["horizontes"][1] + 1)
ULTIMA = pd.Timestamp(PLACEBO["ultima_origem"])
placebos = []
for dep, choque, _ in lpn["lineares"]:
    nome = seca(choque)
    base[f"placebo_{nome}"] = base[nome].shift(-PLACEBO["avanco_meses"])
    irf = lp.estimar(base, especificacao(dep, [f"placebo_{nome}"], horizontes=H_PLACEBO), INICIO,
                     FIM, ultima_origem=ULTIMA)  # fmt: skip
    dp = base.loc[na_janela, nome].std()  # mesma escala da resposta principal
    for c in ["coef", "li_90", "ls_90"]:
        irf[c] = irf[c] * dp
    placebos.append(irf.assign(familia="linear", dependente=dep, chave=choque, tipo="seca"))

# %% [markdown]
# ## Não lineares (Parte 6a): frações em seca, excesso e calor
#
# Frações recalculadas com os limiares alternativos e com os pesos fixos. Antes, a função
# nova reproduz as frações da Parte 6a (limiar ±1,5, pesos defasados).

# %%
nl = cfg["nao_linear"]
TIPOS = ["seca", "excesso", "calor"]
limiares = config["choques"]
referencia = config["features"]["spi"]["periodo_referencia"]
features = lakehouse.ler("features_clima_mensal", config)
pontos = lakehouse.ler("clima_pontos", config)
pesos = {"defasados": lakehouse.ler("pesos_producao_defasados", config),
         "fixos": lakehouse.ler("pesos_producao_fixos", config)}  # fmt: skip
micro = pontos[pontos["tipo"] == "microrregiao"][["codigo", "celula_id"]]
ipca = lakehouse.ler("ipca", config)
entrada = ipca.groupby("area_id")["mes"].min()
inicio_amostra = pd.Timestamp(config["periodos"]["amostra_estimacao"]["inicio"])
areas_longas = [a for a in entrada.index if entrada[a] <= inicio_amostra and a != "brasil"]
sedes = pontos[pontos["tipo"] == "area_ipca"].assign(
    area_id=lambda d: d["ponto_id"].str.removeprefix("area_"))  # fmt: skip
sedes = sedes[sedes["area_id"].isin(areas_longas)]
celulas = pontos.drop_duplicates("celula_id")[["celula_id", "lat_celula", "lon_celula"]]
selecao = cinturao_verde.celulas_no_raio(sedes, celulas, [config["eda"]["raio_cinturao_km"]])


def fracoes(limiar: float, versao_pesos: str) -> pd.DataFrame:
    """Frações por motor e SPI com limiar ±`limiar` (calor fixo) e os pesos indicados."""
    partes = []
    for spi in sorted({m["spi"] for m in nl["modelos"]}):
        motores = {m["motor"] for m in nl["modelos"] if m["spi"] == spi}
        ind = extremos.indicadores_limiar(features, spi, -limiar, limiar, nl["limiar_calor_z"],
                                          (referencia["inicio"], referencia["fim"]))  # fmt: skip
        sel = selecao if "cinturao" in motores and versao_pesos == "defasados" else None
        fr = extremos.fracoes_em_extremo(ind, micro, pesos[versao_pesos],
                                         sorted(motores - {"cinturao"}), sel, TIPOS)  # fmt: skip
        partes.append(fr.assign(spi=spi))
    return pd.concat(partes, ignore_index=True)


principal_fr = fracoes(limiares["spi_excesso"], "defasados")
original = lakehouse.ler("fracoes_extremos_mensal", config)
comparar = principal_fr.merge(original, on=["motor", "spi", "mes"], suffixes=("", "_6a"))
diferenca = max((comparar[t] - comparar[f"{t}_6a"]).abs().max() for t in TIPOS)
print(f"Frações recalculadas × Parte 6a: diferença máxima {diferenca:.2e} "
      f"({len(comparar)} de {len(original)} linhas)")  # fmt: skip
if diferenca > 1e-12 or len(comparar) != len(original):
    raise ValueError("as frações recalculadas não reproduzem a Parte 6a")

VERSOES_FRACAO = {"principal": principal_fr, "pesos_fixos": fracoes(limiares["spi_excesso"],
                                                                      "fixos")}  # fmt: skip
for limiar in testes_cfg["limiares_spi"]:
    VERSOES_FRACAO[f"limiar_{limiar:.1f}"] = fracoes(limiar, "defasados")
for versao, fr in VERSOES_FRACAO.items():
    largo = fr.pivot_table(index="mes", columns=["motor", "spi"], values=TIPOS)
    for tipo, motor, spi in largo.columns:
        base[f"fracao_{tipo}_{motor}_{spi}_{versao}"] = largo[(tipo, motor, spi)]

LIMIARES_ALT = [f"limiar_{v:.1f}" for v in testes_cfg["limiares_spi"]]
TESTES_NAO_LINEARES = ["principal", "com_controles", "sem_2020_2022", "janela_2006",
                       "pesos_fixos", *LIMIARES_ALT]  # fmt: skip
escala = nl["escala_fracao"]
irfs_nl = []
for m in nl["modelos"]:
    dep, motor, spi, h = m["dependente"], m["motor"], m["spi"], m["h_teste"]
    for teste in TESTES_NAO_LINEARES:
        versao = teste if teste in VERSOES_FRACAO else "principal"
        if versao == "pesos_fixos" and motor == "cinturao":
            continue
        nomes = {t: f"fracao_{t}_{motor}_{spi}_{versao}" for t in TIPOS}
        contrastes = [(nomes[a], nomes[b]) for a, b in nl["contrastes"]]
        espec = especificacao(dep, list(nomes.values()), teste == "com_controles")
        espec.contrastes = contrastes
        inicio = JANELAS.get(teste, INICIO)
        extra = {"excluir_periodo": EXCLUIR} if teste == "sem_2020_2022" else {}
        irf = lp.estimar(base, espec, inicio, FIM, **extra)
        tipo = {v: k for k, v in nomes.items()} | {
            f"{nomes[a]} − {nomes[b]}": f"{a} − {b}" for a, b in nl["contrastes"]}  # fmt: skip
        for c in ["coef", "li_90", "ls_90"]:
            irf[c] = irf[c] * escala
        irfs_nl.append(irf.assign(familia="nao_linear", dependente=dep, chave=motor,
                                  tipo=irf["choque"].map(tipo), h_teste=h,
                                  teste=teste))  # fmt: skip
    nomes = {t: f"fracao_{t}_{motor}_{spi}_principal" for t in TIPOS}
    for nome in nomes.values():
        base[f"placebo_{nome}"] = base[nome].shift(-PLACEBO["avanco_meses"])
    espec = especificacao(dep, [f"placebo_{n}" for n in nomes.values()], horizontes=H_PLACEBO)
    irf = lp.estimar(base, espec, INICIO, FIM, ultima_origem=ULTIMA)
    for c in ["coef", "li_90", "ls_90"]:
        irf[c] = irf[c] * escala
    tipo = {f"placebo_{v}": k for k, v in nomes.items()}
    placebos.append(irf.assign(familia="nao_linear", dependente=dep, chave=motor,
                               tipo=irf["choque"].map(tipo)))  # fmt: skip
irfs = pd.concat([irfs_lineares, *irfs_nl], ignore_index=True)
# gzip com data fixa no cabeçalho: o arquivo sai idêntico a cada execução.
irfs.to_csv(TABELAS / "robustez_lp_irfs.csv.gz", index=False,
            compression={"method": "gzip", "mtime": 0})  # fmt: skip
lakehouse.salvar(irfs, "robustez_lp_irfs", config)

# %% [markdown]
# ## Estabilidade no horizonte de teste e classificação

# %%
CHAVES = ["familia", "dependente", "chave", "tipo"]
no_h = irfs[irfs["horizonte"] == irfs["h_teste"]].copy()
no_h["signif"] = (no_h["li_90"] > 0) | (no_h["ls_90"] < 0)
referencia_h = no_h[no_h["teste"] == "principal"].set_index(CHAVES)
no_h = no_h.join(referencia_h[["coef", "signif"]], on=CHAVES, rsuffix="_principal")
no_h["razao"] = no_h["coef"] / no_h["coef_principal"]
no_h["mesmo_sinal"] = np.sign(no_h["coef"]) == np.sign(no_h["coef_principal"])
limite_mag = lpn["classificacao"]["mudanca_magnitude"]


def estado(linha: pd.Series) -> str:
    """Estado do teste em relação à principal (sem a magnitude, que vai à parte)."""
    if linha["teste"] == "principal":
        return "referência"
    if not linha["mesmo_sinal"] and linha["signif"]:
        return "inverte"
    if linha["signif_principal"] and not linha["signif"]:
        return "perde significância"
    if not linha["signif_principal"] and linha["signif"]:
        return "ganha significância"
    return "mantém"


no_h["estado"] = no_h.apply(estado, axis=1)
no_h["magnitude_muda"] = (no_h["teste"] != "principal") & ((no_h["razao"] - 1).abs() > limite_mag)

# Placebo: horizontes 0 a 11 com banda de 90% fora de zero.
placebos = pd.concat(placebos, ignore_index=True)
placebos["signif"] = (placebos["li_90"] > 0) | (placebos["ls_90"] < 0)
placebo = placebos.groupby(CHAVES).agg(placebo_horizontes_signif=("signif", "sum"),
                                       placebo_n=("n", "min")).reset_index()  # fmt: skip
placebo["placebo_falha"] = placebo["placebo_horizontes_signif"] >= PLACEBO["falha_min"]
placebos.to_csv(TABELAS / "robustez_lp_placebo.csv", index=False)

fracao_robusto = lpn["classificacao"]["fracao_significancia_robusto"]
classes = []
for chave, grupo in no_h.groupby(CHAVES, sort=False):
    alternativas = grupo[grupo["teste"] != "principal"]
    ref = grupo[grupo["teste"] == "principal"].iloc[0]
    if ref["signif"]:
        if (alternativas["estado"] == "inverte").any():
            classe = "instável"
        elif (alternativas["signif"] & alternativas["mesmo_sinal"]).mean() >= fracao_robusto:
            classe = "robusto"
        else:
            classe = "frágil"
    else:
        classe = "nulo instável" if alternativas["signif"].any() else "nulo estável"
    mudancas = [f"{t}: {e}" for t, e in zip(alternativas["teste"], alternativas["estado"],
                                           strict=True) if e != "mantém"]  # fmt: skip
    mudancas += [f"{t}: magnitude ×{visual.numero(r)}" for t, r, mm in zip(
        alternativas["teste"], alternativas["razao"], alternativas["magnitude_muda"],
        strict=True) if mm]  # fmt: skip
    classes.append(dict(zip(CHAVES, chave, strict=True)) | {
        "h_teste": int(ref["h_teste"]), "coef_principal": ref["coef"],
        "li_90_principal": ref["li_90"], "ls_90_principal": ref["ls_90"],
        "signif_principal": bool(ref["signif"]), "n_testes": len(alternativas),
        "testes_signif_mesmo_sinal": int(
            (alternativas["signif"] & alternativas["mesmo_sinal"]).sum()),
        "razao_min": alternativas["razao"].min(), "razao_max": alternativas["razao"].max(),
        "classe": classe, "mudancas_relevantes": "; ".join(mudancas),
    })  # fmt: skip
classificacao = pd.DataFrame(classes).merge(placebo, on=CHAVES, how="left")
classificacao.to_csv(TABELAS / "robustez_lp_classificacao.csv", index=False)
estabilidade = no_h[CHAVES + ["h_teste", "teste", "coef", "li_90", "ls_90", "p_valor", "n",
                              "signif", "razao", "estado", "magnitude_muda"]]  # fmt: skip
estabilidade.to_csv(TABELAS / "robustez_lp_estabilidade.csv", index=False)
lakehouse.salvar(estabilidade, "robustez_lp_estabilidade", config)

print(classificacao.drop(columns="mudancas_relevantes").round(2).to_string(index=False))
print(classificacao.groupby(["familia", "classe"]).size())


# %% [markdown]
# ## Diagnóstico pós-teste do placebo (não muda a classificação)
#
# Registrado depois de ver que o placebo falha no canal da ração: o mesmo placebo (a) sem as
# origens cuja janela cruza 2020-2022 e (b) com as secas passadas (t − 1 a t − 13) no modelo.

# %%
diagnosticos = []
for dep, choque, _ in lpn["lineares"]:
    nome = seca(choque)
    futuro = f"placebo_{nome}"
    passado = f"passado_{nome}"
    base[passado] = base[nome].shift(1)
    for versao in lpn["diagnostico_placebo"]:
        choques_modelo = [futuro] if versao == "sem_periodo" else [futuro, passado]
        extra = {"excluir_periodo": EXCLUIR} if versao == "sem_periodo" else {}
        irf = lp.estimar(base, especificacao(dep, choques_modelo, horizontes=H_PLACEBO), INICIO,
                         FIM, ultima_origem=ULTIMA, **extra)  # fmt: skip
        irf = irf[irf["choque"] == futuro].copy()
        dp = base.loc[na_janela, nome].std()
        for c in ["coef", "li_90", "ls_90"]:
            irf[c] = irf[c] * dp
        diagnosticos.append(irf.assign(dependente=dep, chave=choque, versao=versao))
diagnosticos = pd.concat(diagnosticos, ignore_index=True)
diagnosticos["signif"] = (diagnosticos["li_90"] > 0) | (diagnosticos["ls_90"] < 0)
diagnosticos.to_csv(TABELAS / "robustez_lp_placebo_diagnostico.csv", index=False)
contagem = diagnosticos.pivot_table(index=["dependente", "chave"], columns="versao",
                                    values="signif", aggfunc="sum", sort=False)  # fmt: skip
contagem["placebo_registrado"] = classificacao[classificacao["familia"] == "linear"].set_index(
    ["dependente", "chave"]
)["placebo_horizontes_signif"]
print(contagem.to_string())


# %% [markdown]
# ## Figuras: matriz de estabilidade e placebo

# %%
NOMES_CHOQUE = {
    "cwi_milho_spi_6": "milho", "cwi_soja_spi_6": "soja", "cinturao_spi_3": "cinturão",
    "cwi_banana_spi_3": "banana", "cwi_feijao_spi_3": "feijão",
    "cwi_milho_2a_safra_spi_6": "milho 2ª safra", "cinturao": "cinturão", "milho": "milho",
    "banana": "banana", "feijao": "feijão",
}  # fmt: skip
ROTULOS_TESTES = {
    "principal": "principal", "com_controles": "com controles", "sem_2020_2022": "sem 2020–22",
    "janela_2006": "desde 2006-07", "janela_2012": "desde 2012-01",
    "spi_alternativo": "SPI-3 × SPI-6", "pesos_fixos": "pesos fixos",
    "limiar_1.0": "limiar ±1,0", "limiar_2.0": "limiar ±2,0", "placebo": "placebo (t + 12)",
}  # fmt: skip


def chave_linha(linha: pd.Series) -> str:
    return "|".join(str(linha[c]) for c in CHAVES)


def matriz(familia: str, tipos: list[str], testes: list[str]) -> tuple[pd.DataFrame, list, dict]:
    dados = estabilidade[(estabilidade["familia"] == familia)
                         & estabilidade["tipo"].isin(tipos)].copy()  # fmt: skip
    dados["linha"] = dados.apply(chave_linha, axis=1)
    dados["coluna"] = dados["teste"]
    dados["texto"] = dados["coef"].map(visual.numero)
    pl = classificacao[(classificacao["familia"] == familia) & classificacao["tipo"].isin(tipos)]
    pl = pl.dropna(subset=["placebo_falha"])
    contagem_placebo = pl["placebo_horizontes_signif"].astype(int).astype(str) + "/12"
    pl = pl.assign(linha=pl.apply(chave_linha, axis=1), coluna="placebo", texto=contagem_placebo,
                   estado=np.where(pl["placebo_falha"], "placebo falha",
                                   "placebo passa"))  # fmt: skip
    tabela = pd.concat([dados[["linha", "coluna", "estado", "texto"]],
                        pl[["linha", "coluna", "estado", "texto"]]], ignore_index=True)  # fmt: skip
    ordem = list(dict.fromkeys(dados["linha"]))  # ordem do settings
    rotulos = {}
    for chave in ordem:
        _, dep, ch, tipo = chave.split("|")
        h = int(dados.loc[dados["linha"] == chave, "h_teste"].iloc[0])
        sufixo = "" if familia == "linear" else f": {tipo}"
        rotulos[chave] = f"{visual.NOMES_GRUPOS[dep]} × {NOMES_CHOQUE[ch]}{sufixo} (h = {h})"
    return tabela, ordem, rotulos


tabela_lin, ordem_lin, rot_lin = matriz("linear", ["seca"], TESTES_LINEARES)
TIPOS_MATRIZ = ["seca", "excesso", "calor", "seca − excesso"]
tabela_nl, ordem_nl, rot_nl = matriz("nao_linear", TIPOS_MATRIZ, TESTES_NAO_LINEARES)
tabela_matriz = pd.concat([tabela_lin.assign(familia="linear"),
                           tabela_nl.assign(familia="nao_linear")], ignore_index=True)  # fmt: skip
tabela_matriz.to_csv(TABELAS / "robustez_lp_matriz.csv", index=False)
fig = visual.matriz_estabilidade(
    tabela_lin, ordem_lin, [*TESTES_LINEARES, "placebo"], rot_lin, ROTULOS_TESTES,
    titulo="Estabilidade das respostas à seca (Local Projections lineares)",
    nota=("Resposta acumulada do preço (p.p.) por 1 desvio-padrão de seca (−SPI) no horizonte de "
          "teste de cada grupo; cor = estado em relação à principal (banda de 90%). Placebo: "
          "horizontes de 0 a 11 com banda fora de zero quando o choque é o de t + 12 (falha com 3 "
          "ou mais; acaso: 1,2). Tabelas: robustez_lp_estabilidade.csv e "
          "robustez_lp_classificacao.csv."),
)  # fmt: skip
fig.savefig(FIGURAS / "robustez_lp_lineares.png", dpi=150)
fig = visual.matriz_estabilidade(
    tabela_nl, ordem_nl, [*TESTES_NAO_LINEARES, "placebo"], rot_nl, ROTULOS_TESTES,
    titulo="Estabilidade das respostas a extremos (frações em seca, excesso e calor)",
    nota=("Resposta acumulada do preço (p.p.) por +10 p.p. de fração em condição extrema, no "
          "horizonte de teste; cor = estado em relação à principal (banda de 90%). Contrastes sem "
          "placebo. Tabelas: robustez_lp_estabilidade.csv e robustez_lp_classificacao.csv."),
)  # fmt: skip
fig.savefig(FIGURAS / "robustez_lp_nao_lineares.png", dpi=150)

# Placebo × efeito estimado (h = 0 a 11), nos coeficientes centrais das perguntas 1 e 3.
CASOS_PLACEBO = [
    ("linear", "alimentacao_domicilio", "cwi_milho_spi_6", "seca"),
    ("linear", "aves_ovos", "cwi_milho_spi_6", "seca"),
    ("linear", "carnes", "cwi_milho_spi_6", "seca"),
    ("linear", "tuberculos_raizes_legumes", "cinturao_spi_3", "seca"),
    ("nao_linear", "aves_ovos", "milho", "seca"),
    ("nao_linear", "carnes", "milho", "seca"),
    ("nao_linear", "tuberculos_raizes_legumes", "cinturao", "excesso"),
    ("nao_linear", "hortalicas_verduras", "cinturao", "excesso"),
]
partes = []
for caso in CASOS_PLACEBO:
    filtro = lambda d, c=caso: (d[CHAVES] == pd.Series(c, index=CHAVES)).all(axis=1)  # noqa: E731
    est = irfs[filtro(irfs) & (irfs["teste"] == "principal") & irfs["horizonte"].isin(H_PLACEBO)]
    pla = placebos[filtro(placebos)]
    painel = "|".join(caso)
    partes += [est.assign(painel=painel, serie="estimado"),
               pla.assign(painel=painel, serie="placebo")]  # fmt: skip
figura_placebo = pd.concat(partes, ignore_index=True)[
    ["painel", "serie", "horizonte", "coef", "li_90", "ls_90", "n"]]  # fmt: skip
figura_placebo.to_csv(TABELAS / "robustez_lp_placebo_figura.csv", index=False)
CURTOS = {"alimentacao_domicilio": "Alimentação no domicílio", "aves_ovos": "Aves e ovos",
          "carnes": "Carnes", "tuberculos_raizes_legumes": "Tubérculos",
          "hortalicas_verduras": "Hortaliças"}  # fmt: skip
EXTREMO = {"seca": "seca extrema", "excesso": "excesso extremo"}
TITULOS_PLACEBO = {
    "|".join(c): (f"{CURTOS[c[1]]} × {NOMES_CHOQUE[c[2]]}"
                  + ("" if c[0] == "linear" else f" ({EXTREMO[c[3]]})"))
    for c in CASOS_PLACEBO
}  # fmt: skip
fig = visual.placebo_paineis(
    figura_placebo, list(TITULOS_PLACEBO), TITULOS_PLACEBO,
    {"estimado": "choque em t (estimado)", "placebo": "choque em t + 12 (placebo)"},
    titulo="Falsificação: um choque que ainda não aconteceu não deveria mexer no preço",
    nota=("Resposta acumulada do preço (p.p.) de h = 0 a 11 ao choque em t (principal) e ao "
          "choque de t + 12 no lugar dele (placebo, origens até 2022-06); faixas = bandas HAC de "
          "90%. 1ª linha: seca linear (por 1 desvio-padrão); 2ª linha: frações em condição "
          "extrema (por +10 p.p.). Tabela: robustez_lp_placebo_figura.csv."),
    rotulo_y="variação acumulada do preço (p.p.)",
)  # fmt: skip
fig.savefig(FIGURAS / "robustez_lp_placebo.png", dpi=150)


# %% [markdown]
# # Parte 8b — Áreas do IPCA (6b) e janela regional da soja
#
# A base por área é reconstruída para qualquer raio e conjunto de áreas; antes dos testes, a
# versão de 150 km com as 11 áreas precisa reproduzir a base e um painel da 6b.

# %%
ar = cfg["areas"]
a8 = rob["areas_8b"]
AREAS, DEPS = ar["areas"], ar["dependentes"]
ESC = ar["choques"]
CHOQUES_AREA = ["seca_local", "seca_centro_oeste", "seca_matopiba"]
H_TESTE = ar["h_teste"]
todas_sedes = pontos[pontos["tipo"] == "area_ipca"].assign(
    area_id=lambda d: d["ponto_id"].str.removeprefix("area_"))  # fmt: skip
cv_mensal = lakehouse.ler("cinturao_verde_mensal", config)
cwi_def = lakehouse.ler("cwi_mensal", config).query("versao == 'defasado'")
indicadores_spi3 = extremos.indicadores_limiar(
    features, "spi_3", limiares["spi_seca"], limiares["spi_excesso"], nl["limiar_calor_z"],
    (referencia["inicio"], referencia["fim"]))  # fmt: skip


def base_areas_para(areas: list[str], raio: float) -> pd.DataFrame:
    """Base longa por área (como a da 6b) para o raio e as áreas pedidos."""
    precos = ipca[ipca["area_id"].isin(areas) & ipca["categoria"].isin(DEPS)]
    precos = precos[(precos["mes"] >= inicio_amostra) & (precos["mes"] <= FIM)]
    dados = precos.assign(valor=np.log(precos["indice"])).pivot_table(
        index=["area_id", "mes"], columns="categoria", values="valor")  # fmt: skip
    dados.columns = [f"log_{c}" for c in dados.columns]
    dados = dados.reset_index()
    local = cv_mensal[(cv_mensal["raio_km"] == raio) & cv_mensal["area_id"].isin(areas)]
    dados = dados.merge(local[["area_id", "mes", ESC["local"]["spi"]]].rename(
        columns={ESC["local"]["spi"]: "local"}), on=["area_id", "mes"], how="left")  # fmt: skip
    for nome in ["centro_oeste", "matopiba"]:
        c = ESC[nome]
        filtro = (cwi_def["regiao"] == c["regiao"]) & (cwi_def["cultura"] == c["cultura"])
        serie = cwi_def[filtro].set_index("mes")[c["spi"]].rename(nome)
        dados = dados.merge(serie, left_on="mes", right_index=True, how="left")
    sel = cinturao_verde.celulas_no_raio(todas_sedes[todas_sedes["area_id"].isin(areas)],
                                         celulas, [raio])  # fmt: skip
    fracao = cinturao_verde.agregar(indicadores_spi3, sel, TIPOS)
    fracao = fracao[["area_id", "mes", *TIPOS]].rename(
        columns={t: f"fracao_{t}_local" for t in TIPOS})  # fmt: skip
    dados = dados.merge(fracao, on=["area_id", "mes"], how="left")
    janela = (dados["mes"] >= INICIO) & (dados["mes"] <= FIM)
    for bruto in ["local", "centro_oeste", "matopiba"]:
        dp = dados[janela].groupby("area_id")[bruto].std()
        dados[f"seca_{bruto}"] = SINAL * dados[bruto] / dados["area_id"].map(dp)
    return dados.sort_values(["area_id", "mes"]).reset_index(drop=True)


base_150 = base_areas_para(AREAS, ESC["local"]["raio_km"])
original_6b = lakehouse.ler("base_analise_areas", config)
colunas_6b = ["seca_local", "seca_centro_oeste", "seca_matopiba",
              *[f"fracao_{t}_local" for t in TIPOS], *[f"log_{d}" for d in DEPS]]  # fmt: skip
junto = base_150.merge(original_6b, on=["area_id", "mes"], suffixes=("", "_6b"))
diferenca = max((junto[c] - junto[f"{c}_6b"]).abs().max() for c in colunas_6b)
print(f"Base por área (150 km, 11 áreas) × 6b: diferença máxima {diferenca:.1e} "
      f"({len(junto)} de {len(original_6b)} linhas)")  # fmt: skip
if diferenca > 1e-9 or len(junto) != len(original_6b):
    raise ValueError("a base por área reconstruída não reproduz a da 6b")


def espec_area(dep: str, choques: list[str], contrastes=(), horizontes=HORIZONTES):
    e = especificacao(dep, choques, horizontes=horizontes)
    e.contrastes = list(contrastes)
    return e


CONTRASTES_AREA = [("seca_local", "seca_centro_oeste"), ("seca_local", "seca_matopiba")]
NOMES_NL_AREA = [f"fracao_{t}_local" for t in TIPOS]
CONTRASTES_NL_AREA = [(f"fracao_{a}_local", f"fracao_{b}_local") for a, b in nl["contrastes"]]


def paineis(dados: pd.DataFrame, tipos_painel: list[str], **extra) -> pd.DataFrame:
    """Painéis da 6b (especificação principal) nos dados indicados; coeficientes na escala
    publicada (frações por +10 p.p.; choques padronizados por 1 desvio-padrão)."""
    partes = []
    for dep in DEPS:
        for tp in tipos_painel:
            if tp == "efeito_tempo_linear":
                espec, tempo = espec_area(dep, ["seca_local"]), True
            elif tp == "efeito_tempo_nao_linear":
                espec, tempo = espec_area(dep, NOMES_NL_AREA, CONTRASTES_NL_AREA), True
            else:
                espec, tempo = espec_area(dep, CHOQUES_AREA, CONTRASTES_AREA), False
            if "horizontes" in extra:
                espec.horizontes = extra["horizontes"]
            opcoes = {k: v for k, v in extra.items() if k != "horizontes"}
            irf = lp.estimar_painel(dados, espec, INICIO, FIM, efeito_tempo=tempo, **opcoes)
            fracao = irf["choque"].str.startswith("fracao_")
            for c in ["coef", "li_90", "ls_90"]:
                irf[c] = irf[c].where(~fracao, irf[c] * nl["escala_fracao"])
            partes.append(irf.assign(painel=tp, dependente=dep))
    return pd.concat(partes, ignore_index=True)


TIPOS_PAINEL = ["efeito_tempo_linear", "efeito_tempo_nao_linear", "sem_efeito_tempo"]
painel_150 = paineis(base_150, ["efeito_tempo_linear"])
lp_painel_6b = pd.read_csv(TABELAS / "lp_painel.csv")
ref_6b = lp_painel_6b[(lp_painel_6b["especificacao"] == "principal")]
conferencia = painel_150.merge(ref_6b, on=["painel", "dependente", "choque", "horizonte"],
                               suffixes=("", "_6b"))  # fmt: skip
dif_painel = (conferencia["coef"] - conferencia["coef_esc"]).abs().max()
print(f"Painel linear com efeito de tempo × 6b: diferença máxima {dif_painel:.1e}")
if dif_painel > 1e-9:
    raise ValueError("o painel reconstruído não reproduz a 6b")

# %% [markdown]
# ## Testes da 8b: cinturão de 300 km, 16 áreas e placebo nos painéis

# %%
base_300 = base_areas_para(AREAS, 300.0)
areas_16 = sorted(todas_sedes["area_id"])
base_16 = base_areas_para(areas_16, ESC["local"]["raio_km"])
print("Áreas e início (16 áreas):",
      base_16.groupby("area_id")["mes"].min().dt.strftime("%Y-%m").to_dict())  # fmt: skip

# Por área, 300 km: contagem de áreas com banda fora de zero em h_teste (principal).
partes = []
for area in AREAS:
    dados = base_300[base_300["area_id"] == area].set_index("mes")
    for dep in DEPS:
        irf = lp.estimar(dados, espec_area(dep, CHOQUES_AREA, CONTRASTES_AREA), INICIO, FIM)
        partes.append(irf.assign(area_id=area, dependente=dep))
areas_300 = pd.concat(partes, ignore_index=True)
areas_300 = areas_300[areas_300["horizonte"] == areas_300["dependente"].map(H_TESTE)]
contagem_300 = (
    areas_300.assign(acima=areas_300["li_90"] > 0, abaixo=areas_300["ls_90"] < 0)
    .groupby(["dependente", "choque"])[["acima", "abaixo"]].sum().reset_index()
)  # fmt: skip
contagem_6b = pd.read_csv(TABELAS / "lp_areas_contagem.csv").query("especificacao == 'principal'")
contagem_areas = contagem_300.merge(
    contagem_6b[["dependente", "choque", "acima", "abaixo"]], on=["dependente", "choque"],
    suffixes=("_300km", "_150km"))  # fmt: skip
contagem_areas.to_csv(TABELAS / "robustez_areas_contagem.csv", index=False)
print(contagem_areas.to_string(index=False))

testes_paineis = {
    "raio_300": paineis(base_300, TIPOS_PAINEL),
    "painel_16_areas": paineis(base_16, TIPOS_PAINEL[:2], descartar_incompletos=True),
}
principal_paineis = ref_6b.assign(coef=ref_6b["coef_esc"], li_90=ref_6b["li_90_esc"],
                                  ls_90=ref_6b["ls_90_esc"], teste="principal")  # fmt: skip
todos_paineis = pd.concat([principal_paineis[["painel", "dependente", "choque", "horizonte",
                                              "coef", "li_90", "ls_90", "n", "teste"]],
                           *[t.assign(teste=k) for k, t in testes_paineis.items()]],
                          ignore_index=True)  # fmt: skip
todos_paineis["h_teste"] = todos_paineis["dependente"].map(H_TESTE)

# Placebo: choque local de t + 12 (dentro de cada área), origens até 2022-06, h = 0 a 11.
base_placebo = base_150.copy()
for coluna in ["seca_local", *NOMES_NL_AREA]:
    base_placebo[coluna] = base_placebo.groupby("area_id")[coluna].shift(
        -PLACEBO["avanco_meses"])  # fmt: skip
placebo_paineis = paineis(base_placebo, TIPOS_PAINEL[:2], horizontes=H_PLACEBO,
                          ultima_origem=ULTIMA)  # fmt: skip
placebo_paineis["signif"] = (placebo_paineis["li_90"] > 0) | (placebo_paineis["ls_90"] < 0)
placebo_paineis.to_csv(TABELAS / "robustez_paineis_placebo.csv", index=False)

# %% [markdown]
# ## Estabilidade dos painéis no horizonte de teste (regra da 8a)

# %%
CHAVES_P = ["painel", "dependente", "choque"]
no_h_p = todos_paineis[todos_paineis["horizonte"] == todos_paineis["h_teste"]].copy()
no_h_p["signif"] = (no_h_p["li_90"] > 0) | (no_h_p["ls_90"] < 0)
ref_p = no_h_p[no_h_p["teste"] == "principal"].set_index(CHAVES_P)[["coef", "signif"]]
no_h_p = no_h_p.join(ref_p, on=CHAVES_P, rsuffix="_principal")
no_h_p["razao"] = no_h_p["coef"] / no_h_p["coef_principal"]
no_h_p["mesmo_sinal"] = np.sign(no_h_p["coef"]) == np.sign(no_h_p["coef_principal"])
no_h_p["estado"] = no_h_p.apply(estado, axis=1)
no_h_p["magnitude_muda"] = (no_h_p["teste"] != "principal") & (
    (no_h_p["razao"] - 1).abs() > limite_mag)  # fmt: skip
contagem_placebo_p = (placebo_paineis.groupby(CHAVES_P)
                      .agg(placebo_horizontes_signif=("signif", "sum")).reset_index())  # fmt: skip
contagem_placebo_p["placebo_falha"] = (
    contagem_placebo_p["placebo_horizontes_signif"] >= PLACEBO["falha_min"])  # fmt: skip
classes_p = []
for chave, grupo in no_h_p.groupby(CHAVES_P, sort=False):
    ref = grupo[grupo["teste"] == "principal"].iloc[0]
    alternativas = grupo[grupo["teste"] != "principal"]
    mantidos = alternativas["signif"] & alternativas["mesmo_sinal"]
    if ref["signif"]:
        classe = ("instável" if (alternativas["estado"] == "inverte").any()
                  else "robusto" if mantidos.mean() >= fracao_robusto else "frágil")  # fmt: skip
    else:
        classe = "nulo instável" if alternativas["signif"].any() else "nulo estável"
    mudancas = [f"{t}: {e}" for t, e in zip(alternativas["teste"], alternativas["estado"],
                                           strict=True) if e != "mantém"]  # fmt: skip
    mudancas += [f"{t}: magnitude ×{visual.numero(r)}" for t, r, mm in zip(
        alternativas["teste"], alternativas["razao"], alternativas["magnitude_muda"],
        strict=True) if mm]  # fmt: skip
    classes_p.append(dict(zip(CHAVES_P, chave, strict=True)) | {
        "h_teste": int(ref["h_teste"]), "coef_principal": ref["coef"],
        "li_90_principal": ref["li_90"], "ls_90_principal": ref["ls_90"],
        "signif_principal": bool(ref["signif"]), "n_testes": len(alternativas),
        "coef_raio_300": alternativas.set_index("teste")["coef"].get("raio_300"),
        "coef_painel_16_areas": alternativas.set_index("teste")["coef"].get("painel_16_areas"),
        "classe": classe, "mudancas_relevantes": "; ".join(mudancas),
    })  # fmt: skip
classificacao_p = pd.DataFrame(classes_p).merge(contagem_placebo_p, on=CHAVES_P, how="left")
classificacao_p.to_csv(TABELAS / "robustez_paineis_classificacao.csv", index=False)
estab_p = no_h_p[CHAVES_P + ["h_teste", "teste", "coef", "li_90", "ls_90", "n", "signif", "razao",
                             "estado", "magnitude_muda"]]  # fmt: skip
estab_p.to_csv(TABELAS / "robustez_paineis_estabilidade.csv", index=False)
print(classificacao_p.drop(columns="mudancas_relevantes").round(2).to_string(index=False))

# Falsificação aves × calor local ao longo de todos os horizontes do placebo.
aves_calor = placebo_paineis[(placebo_paineis["dependente"] == "aves_ovos")
                             & (placebo_paineis["choque"] == "fracao_calor_local")]  # fmt: skip
print("Placebo aves × calor local (h, coef, signif):",
      aves_calor[["horizonte", "coef", "signif"]].round(3).values.tolist())  # fmt: skip

# %% [markdown]
# ## Soja: validação do CWI com janela crítica por região (critério da Parte 3)

# %%
sj = a8["soja_janela_regional"]
criterio_soja = [c for c in config["espacial"]["validacao_cwi"] if c["cultura"] == "soja"]
anos_validacao = (pd.Timestamp(config["periodos"]["amostra_estimacao"]["inicio"]).year,
                  FIM.year)  # fmt: skip
pam = lakehouse.ler("pam", config)
rendimento = pam[pam["nivel"] == "n1"][["ano", "produto", "rendimento_kg_ha"]]
nacional = cwi_def[(cwi_def["regiao"] == "brasil") & (cwi_def["cultura"] == "soja")]
resumo_nac, tabela_nac = cwi.validar(nacional, rendimento, criterio_soja, anos_validacao)
original_p3 = pd.read_csv(TABELAS / "validacao_cwi.csv").set_index("cultura").loc["soja"]
if (resumo_nac.iloc[0]["correlacao"], resumo_nac.iloc[0]["p_valor"]) != (
        original_p3["correlacao"], original_p3["p_valor"]):  # fmt: skip
    raise ValueError("a validação nacional não reproduz a Parte 3")

micro_uf = pontos[pontos["tipo"] == "microrregiao"][["codigo", "celula_id", "uf"]]
spi_micro = micro_uf.merge(features[["celula_id", "mes", "spi_3"]], on="celula_id")
mes_alvo = np.where(spi_micro["uf"].isin(sj["ufs_sul"]), sj["mes_sul"], sj["mes_demais"])
na_janela_critica = spi_micro[spi_micro["mes"].dt.month == mes_alvo]
# Um valor por microrregião e safra, datado no mês final nacional (fevereiro) para agregar.
indicador = na_janela_critica.assign(
    mes=pd.to_datetime(na_janela_critica["mes"].dt.year.astype(str) + "-02-01"))[
    ["codigo", "mes", "spi_3"]]  # fmt: skip
regional = cwi.cwi(indicador, pesos["defasados"][pesos["defasados"]["cultura"] == "soja"],
                   ["spi_3"])  # fmt: skip
resumo_reg, tabela_reg = cwi.validar(regional, rendimento, criterio_soja, anos_validacao)
peso_sul = pesos["defasados"][(pesos["defasados"]["cultura"] == "soja") & pesos["defasados"][
    "codigo"].isin(micro_uf.loc[micro_uf["uf"].isin(sj["ufs_sul"]), "codigo"])].groupby(
    "ano")["peso"].sum()  # fmt: skip
NOME_NAC, NOME_REG = "soja: dez–fev (Parte 3)", "soja: Sul jan–mar, demais dez–fev"
soja = pd.concat([resumo_nac.assign(cultura=NOME_NAC), resumo_reg.assign(cultura=NOME_REG)],
                 ignore_index=True)  # fmt: skip
soja["peso_sul_medio"] = peso_sul.loc[anos_validacao[0] : anos_validacao[1]].mean()
soja.to_csv(TABELAS / "robustez_soja_janela_regional.csv", index=False)
tabela_soja = pd.concat([tabela_nac.assign(cultura=NOME_NAC), tabela_reg.assign(cultura=NOME_REG)],
                        ignore_index=True)  # fmt: skip
tabela_soja.to_csv(TABELAS / "robustez_soja_safras.csv", index=False)
print(soja.to_string(index=False))
fig = visual.dispersao_cwi_rendimento(
    tabela_soja, soja, "Soja: CWI na janela crítica contra a anomalia do rendimento nacional"
)
fig.savefig(FIGURAS / "robustez_soja_janela_regional.png", dpi=150)

# %% [markdown]
# ## Figura: estabilidade dos painéis da 6b

# %%
NOMES_PAINEL = {"efeito_tempo_linear": "efeito de tempo", "efeito_tempo_nao_linear":
                "efeito de tempo", "sem_efeito_tempo": "sem efeito de tempo"}  # fmt: skip
NOMES_CHOQUE_AREA = {
    "seca_local": "seca local (−SPI)", "fracao_seca_local": "seca extrema local",
    "fracao_excesso_local": "excesso local", "fracao_calor_local": "calor local",
    "seca_centro_oeste": "seca no CO", "seca_matopiba": "seca no Matopiba",
}  # fmt: skip
CURTOS |= {"hortalicas_verduras": "Hortaliças", "tuberculos_raizes_legumes": "Tubérculos"}
linhas_p, rotulos_p = [], {}
for tp, choques_tp in [("efeito_tempo_linear", ["seca_local"]),
                       ("efeito_tempo_nao_linear", NOMES_NL_AREA),
                       ("sem_efeito_tempo", CHOQUES_AREA)]:  # fmt: skip
    for dep in DEPS:
        for choque in choques_tp:
            chave = f"{tp}|{dep}|{choque}"
            linhas_p.append(chave)
            rotulos_p[chave] = (f"{CURTOS[dep]} × {NOMES_CHOQUE_AREA[choque]}, "
                                f"{NOMES_PAINEL[tp]} (h = {H_TESTE[dep]})")  # fmt: skip
celulas = estab_p[estab_p["choque"].isin(NOMES_CHOQUE_AREA)].assign(
    linha=lambda d: d["painel"] + "|" + d["dependente"] + "|" + d["choque"],
    coluna=lambda d: d["teste"], texto=lambda d: d["coef"].map(visual.numero))  # fmt: skip
com_placebo = classificacao_p.dropna(subset=["placebo_falha"])
com_placebo = com_placebo[com_placebo["choque"].isin(NOMES_CHOQUE_AREA)].assign(
    linha=lambda d: d["painel"] + "|" + d["dependente"] + "|" + d["choque"], coluna="placebo",
    estado=lambda d: np.where(d["placebo_falha"], "placebo falha", "placebo passa"),
    texto=lambda d: d["placebo_horizontes_signif"].astype(int).astype(str) + "/12")  # fmt: skip
tabela_matriz_p = pd.concat([celulas[["linha", "coluna", "estado", "texto"]],
                             com_placebo[["linha", "coluna", "estado", "texto"]]],
                            ignore_index=True)  # fmt: skip
tabela_matriz_p.to_csv(TABELAS / "robustez_paineis_matriz.csv", index=False)
fig = visual.matriz_estabilidade(
    tabela_matriz_p, linhas_p, ["principal", "raio_300", "painel_16_areas", "placebo"],
    rotulos_p, {"principal": "6b (150 km, 11 áreas)", "raio_300": "cinturão de 300 km",
                "painel_16_areas": "16 áreas", "placebo": "placebo (t + 12)"},
    titulo="Estabilidade dos painéis por área (pergunta 2)",
    nota=("Painéis das áreas do IPCA (6b): resposta acumulada do preço (p.p.) por 1 desvio-padrão "
          "(choques −SPI) ou por +10 p.p. do cinturão em condição extrema, no horizonte de teste; "
          "cor = estado em relação à 6b (banda de 90%). Efeito de tempo = efeito fixo de cada mês "
          "(só o clima próprio de cada área identifica). 16 áreas só nos painéis com efeito de "
          "tempo; placebo só neles. Tabelas: robustez_paineis_estabilidade.csv e "
          "robustez_paineis_classificacao.csv."),
)  # fmt: skip
fig.savefig(FIGURAS / "robustez_paineis.png", dpi=150)

# %% [markdown]
# # Parte 8c — Robustez da previsão (6c) e consolidação da Parte 8
#
# Variações registradas antes de rodar, uma de cada vez: janela móvel de 120 meses (todos os
# modelos), avaliação sem as origens cuja janela do alvo cruza 2020–2022 e, em hortaliças e
# tubérculos, o clima local como frações do cinturão em seca, excesso e calor. Antes, o caminho
# novo precisa reproduzir as previsões (`enso_previsoes.csv.gz`) e a avaliação da 6c.

# %%
pr = config["modelos"]["previsao"]
en = cfg["enso"]
rp, p8c = rob["previsao"], rob["previsao_8c"]
PRIMEIRA_ORIGEM = pd.Timestamp(pr["primeira_origem"])
NIVEL = pr["nivel_teste"]
INDICES = list(en["indices"])
for nome, indice in en["indices"].items():
    base[f"enso_{nome}_bruto"] = base[indice["coluna"]].shift(indice["defasagem"])
ENSO_PREV = {i: [f"enso_{i}_bruto"] for i in INDICES}
COMPARACOES = {"oni": "benchmark", "mei": "benchmark", "local": "benchmark",
               "local_oni": "local", "local_mei": "local"}  # fmt: skip
mf = p8c["motor_fracoes"]
LOCAL_FRACOES = [f"fracao_{t}_{mf['motor']}_{mf['spi']}_principal" for t in TIPOS]


def rodar_previsoes(deps: list[str], modelos: list[str], local: dict[str, list[str]],
                    janela_movel: int | None = None) -> pd.DataFrame:  # fmt: skip
    """Previsões diretas da 6c (mesma especificação) para os grupos e modelos indicados."""
    partes = []
    for dep in deps:
        preditores = {"benchmark": [], "oni": ENSO_PREV["oni"], "mei": ENSO_PREV["mei"],
                      "local": local[dep], "local_oni": local[dep] + ENSO_PREV["oni"],
                      "local_mei": local[dep] + ENSO_PREV["mei"]}  # fmt: skip
        for modelo in modelos:
            espec = pv.EspecificacaoPrevisao(
                log_preco=f"log_{dep}", preditores=preditores[modelo],
                lags_inflacao=pr["lags_inflacao"], lags_preditor=pr["lags_preditor"],
                dummies_mes=pr["dummies_mes"],
            )  # fmt: skip
            for h in pr["horizontes"]:
                prev = pv.prever(base, espec, h, INICIO, PRIMEIRA_ORIGEM, FIM, janela_movel)
                partes.append(prev.assign(dependente=dep, modelo=modelo))
    previsoes = pd.concat(partes, ignore_index=True)
    previsoes["mes_alvo"] = (
        previsoes["origem"].dt.to_period("M") + previsoes["horizonte"].to_numpy()
    ).dt.to_timestamp()
    return previsoes


# Conferência: o caminho novo reproduz a 6c (previsões, Clark-West e Diebold-Mariano).
prev_principal = rodar_previsoes(en["dependentes"], pr["modelos"], pr["local"])
prev_6c = pd.read_csv(TABELAS / "enso_previsoes.csv.gz", parse_dates=["origem", "mes_alvo"])
CHAVES_PREV = ["dependente", "modelo", "horizonte", "origem"]
juntas = prev_principal.merge(prev_6c, on=CHAVES_PREV, suffixes=("", "_6c"), validate="1:1")
dif_prev = max((juntas[c] - juntas[f"{c}_6c"]).abs().max() for c in ["alvo", "previsao", "erro"])
av_principal, dm_principal = pv.avaliar(prev_principal, COMPARACOES, INDICES, NIVEL)
av_6c = pd.read_csv(TABELAS / "enso_previsao_avaliacao.csv")
dm_6c = pd.read_csv(TABELAS / "enso_previsao_dm.csv")
dif_av = (av_principal[["r2_oos_pct", "cw_p"]] - av_6c[["r2_oos_pct", "cw_p"]]).abs().max().max()
dif_dm = (dm_principal["dm_p"] - dm_6c["dm_p"]).abs().max()
print(f"Reprodução da 6c: {len(juntas)} de {len(prev_6c)} previsões, diferença máxima "
      f"{dif_prev:.1e}; avaliação {dif_av:.1e}; Diebold-Mariano {dif_dm:.1e}; classes iguais: "
      f"{(dm_principal['classe'] == dm_6c['classe']).all()}")  # fmt: skip
if (len(juntas) != len(prev_6c) or max(dif_prev, dif_av, dif_dm) > 1e-8
        or not (dm_principal["classe"] == dm_6c["classe"]).all()):  # fmt: skip
    raise ValueError("o caminho novo não reproduz a previsão da 6c")

# %% [markdown]
# ## As três variações

# %%
prev_movel = rodar_previsoes(en["dependentes"], pr["modelos"], pr["local"],
                             rp["janela_movel_meses"])  # fmt: skip
cruza = pv.alvo_cruza_periodo(prev_principal["origem"], prev_principal["horizonte"], *EXCLUIR)
prev_sem = prev_principal[~cruza.to_numpy()]
GRUPOS_FR = p8c["grupos_fracoes"]
prev_fr_local = rodar_previsoes(GRUPOS_FR, ["local", "local_oni", "local_mei"],
                                {d: LOCAL_FRACOES for d in GRUPOS_FR})  # fmt: skip
prev_fr = pd.concat([prev_principal[prev_principal["dependente"].isin(GRUPOS_FR)
                                    & prev_principal["modelo"].isin(["benchmark", *INDICES])],
                     prev_fr_local], ignore_index=True)  # fmt: skip
VARIANTES = {"principal": prev_principal, "janela_movel": prev_movel,
             "sem_2020_2022": prev_sem, "fracoes_hortifruti": prev_fr}  # fmt: skip
partes_av, partes_dm = [], []
for variante, previsoes in VARIANTES.items():
    av, dm = pv.avaliar(previsoes, COMPARACOES, INDICES, NIVEL)
    partes_av.append(av.assign(variante=variante))
    partes_dm.append(dm.assign(variante=variante))
avaliacao_8c = pd.concat(partes_av, ignore_index=True)
dm_8c = pd.concat(partes_dm, ignore_index=True)
avaliacao_8c.to_csv(TABELAS / "robustez_previsao_avaliacao.csv", index=False)
dm_8c.to_csv(TABELAS / "robustez_previsao_dm.csv", index=False)
novas = pd.concat([prev_movel.assign(variante="janela_movel"),
                   prev_fr_local.assign(variante="fracoes_hortifruti")],
                  ignore_index=True)  # fmt: skip
novas.to_csv(TABELAS / "robustez_previsoes.csv.gz", index=False,
             compression={"method": "gzip", "mtime": 0})  # fmt: skip
for variante, previsoes in VARIANTES.items():
    print(variante, "| origens por h:", previsoes.groupby("horizonte")["origem"].nunique()
          .to_dict(), "| n de estimação:", previsoes["n_estimacao"].min(), "a",
          previsoes["n_estimacao"].max())  # fmt: skip

contagem_8c = (
    avaliacao_8c.dropna(subset=["cw_p"]).groupby(["variante", "modelo"], sort=False)
    .agg(comparado_com=("comparado_com", "first"), casos=("cw_p", "size"),
         vence_10pct=("vence_10pct", "sum"), r2_oos_mediano_pct=("r2_oos_pct", "median"),
         r2_oos_positivo=("r2_oos_pct", lambda s: int((s > 0).sum())))
    .reset_index()
)  # fmt: skip
contagem_8c["esperado_acaso"] = NIVEL * contagem_8c["casos"]
contagem_8c.to_csv(TABELAS / "robustez_previsao_contagem.csv", index=False)
print(contagem_8c.round(2).to_string(index=False))
print(dm_8c.groupby(["variante", "indice", "classe"], sort=False).size().unstack(fill_value=0))

# %% [markdown]
# ## Classificação (regra da 8a; significativo = Clark-West com p < 0,10)

# %%
TESTES_PREV = ["janela_movel", "sem_2020_2022"]
CHAVES_CW = ["dependente", "horizonte", "modelo"]
cw = avaliacao_8c.dropna(subset=["cw_p"])
ref_cw = cw[cw["variante"] == "principal"]
classes_prev = []
for _, ref in ref_cw.iterrows():
    alt = cw[cw["variante"].isin(TESTES_PREV) & (cw["dependente"] == ref["dependente"])
             & (cw["horizonte"] == ref["horizonte"]) & (cw["modelo"] == ref["modelo"])]  # fmt: skip
    if ref["vence_10pct"]:
        classe = "robusto" if alt["vence_10pct"].mean() >= fracao_robusto else "frágil"
    else:
        classe = "nulo instável" if alt["vence_10pct"].any() else "nulo estável"
    mudancas = []
    for _, a in alt.iterrows():
        if ref["vence_10pct"] != a["vence_10pct"]:
            mudancas.append(f"{a['variante']}: {'perde' if ref['vence_10pct'] else 'ganha'} "
                            "vitória")  # fmt: skip
        if np.sign(a["r2_oos_pct"]) != np.sign(ref["r2_oos_pct"]):
            mudancas.append(f"{a['variante']}: R² fora da amostra "
                            f"{visual.numero(ref['r2_oos_pct'], 1)} → "
                            f"{visual.numero(a['r2_oos_pct'], 1)}%")  # fmt: skip
    por_teste = alt.set_index("variante")
    classes_prev.append({c: ref[c] for c in CHAVES_CW} | {
        "comparado_com": ref["comparado_com"], "r2_oos_pct_principal": ref["r2_oos_pct"],
        "cw_p_principal": ref["cw_p"], "vence_principal": bool(ref["vence_10pct"]),
        **{f"r2_oos_pct_{t}": por_teste.loc[t, "r2_oos_pct"] for t in TESTES_PREV},
        **{f"cw_p_{t}": por_teste.loc[t, "cw_p"] for t in TESTES_PREV},
        "classe": classe, "mudancas_relevantes": "; ".join(mudancas),
    })  # fmt: skip
classificacao_prev = pd.DataFrame(classes_prev)
classificacao_prev.to_csv(TABELAS / "robustez_previsao_classificacao.csv", index=False)
print(classificacao_prev.groupby(["modelo", "classe"], sort=False).size().unstack(fill_value=0))
print(classificacao_prev[classificacao_prev["vence_principal"]
                         | (classificacao_prev["classe"] == "nulo instável")]
      .drop(columns="mudancas_relevantes").round(3).to_string(index=False))  # fmt: skip

# Diebold-Mariano: a classe ENSO × local muda?
dm_largo = dm_8c.pivot_table(index=["dependente", "horizonte", "indice"], columns="variante",
                             values="classe", aggfunc="first")  # fmt: skip
for variante in [*TESTES_PREV, "fracoes_hortifruti"]:
    comp = dm_largo.dropna(subset=[variante])
    mudou = comp[comp[variante] != comp["principal"]]
    print(f"DM, {variante}: {len(mudou)} de {len(comp)} classes mudam")
    print(mudou[["principal", variante]].to_string())

# Frações no hortifrúti: o clima local com frações × o SPI-3 linear da 6c.
fr_comp = cw[cw["dependente"].isin(GRUPOS_FR) & cw["modelo"].isin(["local", "local_oni",
                                                                   "local_mei"])
             & cw["variante"].isin(["principal", "fracoes_hortifruti"])]  # fmt: skip
print(fr_comp.pivot_table(index=["dependente", "horizonte", "modelo"], columns="variante",
                          values=["r2_oos_pct", "cw_p"]).round(3).to_string())  # fmt: skip

# %% [markdown]
# ## Consolidação da Parte 8: as classificações da 8a, da 8b e da 8c

# %%
ORDEM_CLASSES = ["robusto", "frágil", "instável", "nulo estável", "nulo instável"]
c8a = classificacao.assign(
    parte="8a", bloco=classificacao["familia"], choque_ou_modelo=classificacao["chave"],
    medida=np.where(classificacao["familia"] == "linear", "p.p. por 1 desvio-padrão de seca",
                    "p.p. por +10 p.p. de fração"),
).rename(columns={"h_teste": "horizonte", "coef_principal": "valor_principal"})  # fmt: skip
c8b = classificacao_p.assign(
    parte="8b", bloco="painel: " + classificacao_p["painel"],
    choque_ou_modelo=classificacao_p["choque"], tipo=np.nan,
    medida=np.where(classificacao_p["choque"].str.startswith("fracao_"),
                    "p.p. por +10 p.p. de fração", "p.p. por 1 desvio-padrão"),
).rename(columns={"h_teste": "horizonte", "coef_principal": "valor_principal"})  # fmt: skip
c8c = classificacao_prev.assign(
    parte="8c", bloco="previsão",
    choque_ou_modelo=classificacao_prev["modelo"] + " × " + classificacao_prev["comparado_com"],
    tipo=np.nan, medida="R² fora da amostra (%)", n_testes=len(TESTES_PREV),
    signif_principal=classificacao_prev["vence_principal"],
).rename(columns={"r2_oos_pct_principal": "valor_principal"})  # fmt: skip
COLUNAS_CONS = ["parte", "bloco", "dependente", "choque_ou_modelo", "tipo", "horizonte", "medida",
                "valor_principal", "signif_principal", "n_testes", "classe",
                "placebo_horizontes_signif", "placebo_falha", "mudancas_relevantes"]  # fmt: skip
consolidada = pd.concat([c[c.columns.intersection(COLUNAS_CONS)] for c in [c8a, c8b, c8c]],
                        ignore_index=True).reindex(columns=COLUNAS_CONS)  # fmt: skip
consolidada.to_csv(TABELAS / "robustez_parte8_consolidada.csv", index=False)
contagem_cons = (consolidada.groupby(["parte", "bloco"], sort=False)["classe"]
                 .value_counts().unstack(fill_value=0)
                 .reindex(columns=ORDEM_CLASSES, fill_value=0))  # fmt: skip
contagem_cons["total"] = contagem_cons.sum(axis=1)
placebo_cons = consolidada.dropna(subset=["placebo_falha"]).groupby(["parte", "bloco"]).agg(
    com_placebo=("placebo_falha", "size"), placebo_falha=("placebo_falha", "sum"))  # fmt: skip
contagem_cons = contagem_cons.join(placebo_cons).fillna({"com_placebo": 0, "placebo_falha": 0})
contagem_cons = contagem_cons.astype(int).reset_index()
contagem_cons.to_csv(TABELAS / "robustez_parte8_contagem.csv", index=False)
print(contagem_cons.to_string(index=False))
print("Total:", contagem_cons[[*ORDEM_CLASSES, "total", "com_placebo", "placebo_falha"]].sum()
      .to_dict())  # fmt: skip

# %% [markdown]
# ## Figura: estabilidade da previsão (pergunta 4)
#
# R² fora da amostra (%) de clima local, MEI e ONI contra o benchmark em cada variação; * =
# vence o benchmark (Clark-West, p < 0,10). Tabelas: `robustez_previsao_avaliacao.csv` e
# `robustez_previsao_classificacao.csv`.

# %%
SERIES_MATRIZ = [("local", "principal"), ("local", "janela_movel"), ("local", "sem_2020_2022"),
                 ("local", "fracoes_hortifruti"), ("mei", "principal"), ("mei", "janela_movel"),
                 ("mei", "sem_2020_2022"), ("oni", "principal"), ("oni", "janela_movel"),
                 ("oni", "sem_2020_2022")]  # fmt: skip
NOMES_VARIANTES = {"principal": "6c", "janela_movel": "janela móvel",
                   "sem_2020_2022": "sem 2020–22", "fracoes_hortifruti": "frações"}  # fmt: skip
NOMES_MODELOS_PREV = {"local": "local", "mei": "MEI", "oni": "ONI"}
celulas_prev = avaliacao_8c[avaliacao_8c["modelo"].isin(NOMES_MODELOS_PREV)].copy()
celulas_prev = celulas_prev.merge(
    celulas_prev[celulas_prev["variante"] == "principal"][CHAVES_CW + ["vence_10pct"]],
    on=CHAVES_CW, suffixes=("", "_principal"),
)  # fmt: skip
celulas_prev["estado"] = np.select(
    [celulas_prev["variante"] == "principal",
     celulas_prev["vence_10pct"] & ~celulas_prev["vence_10pct_principal"],
     ~celulas_prev["vence_10pct"] & celulas_prev["vence_10pct_principal"]],
    ["referência", "ganha significância", "perde significância"], default="mantém",
)  # fmt: skip
celulas_prev = celulas_prev.assign(
    linha=celulas_prev["dependente"] + "|" + celulas_prev["horizonte"].astype(str),
    coluna=celulas_prev["modelo"] + "|" + celulas_prev["variante"],
    texto=[visual.numero(r, 1) + ("*" if v else "")
           for r, v in zip(celulas_prev["r2_oos_pct"], celulas_prev["vence_10pct"], strict=True)],
)  # fmt: skip
tabela_matriz_prev = celulas_prev[["linha", "coluna", "estado", "texto", "r2_oos_pct", "cw_p"]]
tabela_matriz_prev.to_csv(TABELAS / "robustez_previsao_matriz.csv", index=False)
linhas_prev = [f"{d}|{h}" for d in en["dependentes"] for h in pr["horizontes"]]
rotulos_prev = {f"{d}|{h}": f"{visual.NOMES_GRUPOS[d]}, h = {h}" for d in en["dependentes"]
                for h in pr["horizontes"]}  # fmt: skip
colunas_prev = [f"{m}|{v}" for m, v in SERIES_MATRIZ]
fig = visual.matriz_estabilidade(
    tabela_matriz_prev, linhas_prev, colunas_prev, rotulos_prev,
    {f"{m}|{v}": f"{NOMES_MODELOS_PREV[m]}: {NOMES_VARIANTES[v]}" for m, v in SERIES_MATRIZ},
    titulo="Estabilidade da previsão fora da amostra (pergunta 4)",
    nota=("R² fora da amostra (%) = 1 − erro quadrático médio do modelo / o do benchmark "
          "(constante, dummies de mês e inflação dos 12 meses anteriores); * = vence o benchmark "
          "(Clark-West, "
          "p < 0,10). Cor = estado em relação à 6c (expansiva, alvos de 2012-01 a 2023-06). Janela "
          "móvel = estimação com as últimas 120 origens; sem 2020–22 = sem os alvos que cruzam "
          "2020–2022; frações = clima local como frações do cinturão em seca, excesso e calor "
          "(hortaliças e tubérculos; comparado ao local da 6c). Tabelas: "
          "robustez_previsao_avaliacao.csv e robustez_previsao_classificacao.csv."),
)  # fmt: skip
fig.savefig(FIGURAS / "robustez_previsao.png", dpi=150)
