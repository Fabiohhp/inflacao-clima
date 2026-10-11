# %% [markdown]
# # 02 — Engenharia de features climáticas
#
# A partir do clima diário do NASA POWER (por célula da grade), calcula as features mensais
# (chuva, temperatura, ET0, P − ET0, GDD, graus-dia de calor, geada, calor extremo), o SPI-3 e
# o SPI-6 e os extremos nas janelas críticas das culturas. Valida o SPI contra secas
# documentadas, definidas no settings antes do cálculo.
#
# Rodar como script: `python notebooks/02_engenharia_features_climaticas.py [--refresh]`.

# %%
import argparse

import numpy as np
import pandas as pd

from clima_inflacao import lakehouse, visual
from clima_inflacao.config import caminho_projeto, carregar_config, garantir_diretorios
from clima_inflacao.features import agregacao, extremos, spi, validacao
from clima_inflacao.ingestao import malhas, nasa_power
from clima_inflacao.ingestao.cliente_http import total_requisicoes
from clima_inflacao.log import configurar_log

parser = argparse.ArgumentParser(description="Parte 2: features climáticas")
parser.add_argument("--refresh", action="store_true", help="baixa de novo a altitude das células")
args, _ = parser.parse_known_args()

config = carregar_config()
garantir_diretorios(config)
configurar_log(caminho_projeto("logs", config) / "02_features.log")
TABELAS = caminho_projeto("tabelas", config)
FIGURAS = caminho_projeto("figuras", config)
cfg = config["features"]
pd.set_option("display.width", 160, "display.max_columns", 30)

# %% [markdown]
# ## Entradas: clima diário, pontos e altitude das células
#
# A altitude da grade (para o ET0 de Penman-Monteith) vem da geometria de uma consulta curta
# ao NASA POWER por célula, com cache.

# %%
diario = lakehouse.ler("clima_diario", config)
pontos = lakehouse.ler("clima_pontos", config)
celulas = pontos.drop_duplicates("celula_id")[["celula_id", "lat_celula", "lon_celula"]]
celulas = celulas.merge(nasa_power.elevacao(celulas, config, refresh=args.refresh), on="celula_id")
print(f"Células: {len(celulas)}; dias: {len(diario):,}")
print("Altitude (m):", celulas["elevacao_m"].describe().round(0).to_dict())

# %% [markdown]
# ## Features mensais e por safra, em lotes de células
#
# Diário -> mensal: chuva, ET0 e graus-dia somados; Tmax e Tmin em média; geada e calor em
# contagem de dias, mais a maior sequência de calor. O mês só recebe valor se todos os dias
# tiverem dado. As janelas críticas das culturas vêm de `producao.culturas`.

# %%
culturas = config["producao"]["culturas"]
mensais, safras, et0_negativos = [], [], 0
ids = celulas["celula_id"].to_numpy()
for lote in np.array_split(ids, int(np.ceil(len(ids) / 60))):
    derivado = agregacao.diarias_derivadas(diario[diario["celula_id"].isin(lote)], celulas, cfg)
    et0_negativos += int((derivado["et0_pm_mm"] < 0).sum())
    mensais.append(
        agregacao.mensal(derivado, exigir_mes_completo=cfg["agregacao"]["exigir_mes_completo"])
    )
    safras.append(extremos.janelas_criticas(derivado, culturas))
clima_mensal = pd.concat(mensais, ignore_index=True)
clima_safras = pd.concat(safras, ignore_index=True)
del diario, derivado
print(f"Linhas mensais: {len(clima_mensal):,}; linhas por safra: {len(clima_safras):,}")
print(f"Dias com ET0 de Penman-Monteith negativo: {et0_negativos}")

# %% [markdown]
# ## SPI-3 e SPI-6
#
# Gama mista ajustada por célula e mês do ano no período de referência fixo (1991-2020).

# %%
ref = cfg["spi"]["periodo_referencia"]
referencia = (pd.Timestamp(ref["inicio"]), pd.Timestamp(ref["fim"]))
indices_spi = spi.spi_painel(
    clima_mensal[["celula_id", "mes", "precipitacao_mm"]],
    cfg["spi"]["escalas_meses"],
    referencia,
    min_positivos=cfg["spi"]["min_valores_positivos"],
    limite_abs=cfg["spi"]["limite_abs"],
)
features = clima_mensal.merge(indices_spi, on=["celula_id", "mes"], how="left", validate="1:1")
lakehouse.salvar(features, "features_clima_mensal", config)
lakehouse.salvar(clima_safras, "features_clima_safra", config)

# Na referência, o SPI de cada célula deve ter média ~0 e desvio ~1.
colunas_spi = [f"spi_{k}" for k in cfg["spi"]["escalas_meses"]]
na_ref = features[(features["mes"] >= referencia[0]) & (features["mes"] <= referencia[1])]
resumo_ref = na_ref.groupby("celula_id")[colunas_spi].agg(["mean", "std"])
estatisticas_ref = resumo_ref.describe(percentiles=[0.05, 0.5, 0.95]).round(3)
estatisticas_ref.to_csv(TABELAS / "spi_estatisticas_referencia.csv")
print(estatisticas_ref.loc[["min", "5%", "50%", "95%", "max"]])
for coluna in colunas_spi:
    ausentes = features.loc[features["mes"] >= pd.Timestamp("1981-07-01"), coluna].isna().mean()
    truncados = (features[coluna].abs() >= cfg["spi"]["limite_abs"]).mean()
    print(f"{coluna}: {100 * ausentes:.2f}% ausentes; {100 * truncados:.2f}% truncados em ±3,09")

# %% [markdown]
# ## Diagnósticos das demais features

# %%
validos = features.dropna(subset=["et0_pm_mm", "et0_hg_mm"])
razao = validos["et0_hg_mm"].sum() / validos["et0_pm_mm"].sum()
correlacao = validos["et0_pm_mm"].corr(validos["et0_hg_mm"])
print(f"ET0 mensal: Hargreaves / Penman-Monteith = {razao:.3f}; correlação = {correlacao:.3f}")
print(f"ET0 de Penman-Monteith a partir de {validos['mes'].min():%Y-%m} (radiação desde 1984)")

ufs = pontos[pontos["tipo"] == "microrregiao"][["celula_id", "uf"]].drop_duplicates("celula_id")
por_uf = features.merge(ufs, on="celula_id").groupby("uf")
diagnostico = pd.DataFrame(
    {
        "chuva_anual_mm": por_uf["precipitacao_mm"].mean() * 12,
        "et0_pm_anual_mm": por_uf["et0_pm_mm"].mean() * 12,
        "dias_geada_por_ano": por_uf["dias_geada"].mean() * 12,
        "dias_calor_por_ano": por_uf["dias_calor"].mean() * 12,
        "gdd_10_30_anual": por_uf["gdd_10_30"].mean() * 12,
    }
).round(1)
diagnostico.to_csv(TABELAS / "features_diagnostico_por_uf.csv")
print(diagnostico.sort_values("dias_geada_por_ano", ascending=False).to_string())

# %% [markdown]
# ## Critério de aceite: o SPI identifica secas documentadas?
#
# Eventos e critério definidos no settings (`features.validacao_secas`) antes do cálculo: o
# SPI-6 médio das microrregiões das UFs do evento chega a <= -1 em algum mês da janela.

# %%
eventos = cfg["validacao_secas"]
spi_micro = validacao.spi_microrregioes(features, pontos, "spi_6")
resultado_secas, series_eventos = validacao.validar_secas(spi_micro, eventos, "spi_6")
resultado_secas.to_csv(TABELAS / "validacao_secas_spi.csv", index=False)
print(resultado_secas.to_string(index=False))

series_eventos.to_csv(TABELAS / "spi6_series_eventos.csv", index=False)
fig = visual.series_spi_eventos(
    series_eventos, eventos, "spi_medio", pd.Timestamp("2000-01-01"),
    "SPI-6 médio das microrregiões nas regiões das secas documentadas",
)  # fmt: skip
fig.savefig(FIGURAS / "spi6_series_eventos.png")

meses_pico = dict(
    zip(resultado_secas["evento"], pd.to_datetime(resultado_secas["mes_do_minimo"]), strict=True)
)
paineis = [(evento["evento"], meses_pico[evento["evento"]]) for evento in eventos]
mapa_valores = spi_micro[spi_micro["mes"].isin([mes for _, mes in paineis])]
mapa_valores.to_csv(TABELAS / "spi6_mapas_eventos.csv", index=False)
fig = visual.mapas_spi(
    malhas.microrregioes(config), mapa_valores, paineis, "spi_6",
    "SPI-6 por microrregião no mês mais seco de cada evento documentado",
)  # fmt: skip
fig.savefig(FIGURAS / "spi6_mapas_eventos.png")

# %% [markdown]
# ## Homogeneidade do NASA POWER no tempo (contra o INMET)
#
# Viés de chuva e temperatura do NASA POWER contra as 16 estações do INMET da Parte 1c, por
# semestre, relativo ao viés de cada estação em 2010-2021. Uma fonte homogênea fica perto de 1
# (chuva) e de 0 °C (temperatura).

# %%
estacoes_celulas = lakehouse.ler("inmet_estacoes_escolhidas", config)[["codigo"]].merge(
    pontos[pontos["tipo"] == "estacao_inmet"][["codigo", "celula_id"]], on="codigo"
)
homogeneidade = validacao.homogeneidade_nasa_inmet(
    lakehouse.ler("inmet_mensal", config), features, estacoes_celulas, base=(2010, 2021)
)
homogeneidade.to_csv(TABELAS / "homogeneidade_nasa_inmet.csv", index=False)
print(homogeneidade[homogeneidade["semestre"] >= "2016"].round(2).to_string(index=False))

# %%
print(f"Requisições HTTP nesta execução: {total_requisicoes()}")
