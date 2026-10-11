"""Estilo dos gráficos do projeto e figuras reutilizáveis.

Cores da paleta de referência (validadas no modo claro): uma série usa o azul; faixas de
referência usam o cinza neutro; texto nunca usa a cor da série.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SUPERFICIE = "#fcfcfb"
TEXTO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"
GRADE = "#e4e3df"
NEUTRO = "#f0efec"
SERIE_1 = "#2a78d6"


def aplicar_estilo() -> None:
    """Fundo claro, grade e eixos recessivos, texto em tinta neutra."""
    plt.rcParams.update(
        {
            "figure.facecolor": SUPERFICIE,
            "axes.facecolor": SUPERFICIE,
            "savefig.facecolor": SUPERFICIE,
            "axes.edgecolor": GRADE,
            "axes.labelcolor": TEXTO_SECUNDARIO,
            "axes.titlecolor": TEXTO,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRADE,
            "grid.linewidth": 0.6,
            "xtick.color": TEXTO_SECUNDARIO,
            "ytick.color": TEXTO_SECUNDARIO,
            "font.size": 9,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def grafico_cobertura(
    tabela: pd.DataFrame,
    janela: tuple[pd.Timestamp, pd.Timestamp],
    *,
    inicio_eixo: pd.Timestamp,
    titulo: str,
) -> plt.Figure:
    """Linha do tempo: uma barra por série, do início ao fim, com a janela amostral ao fundo.

    `tabela` precisa das colunas linha_figura, inicio, fim e frequencia, na ordem de cima para
    baixo. Séries anuais ("YS") vão até o fim do último ano. Séries que começam antes de
    `inicio_eixo` são cortadas no eixo e marcadas com "◂".
    """
    aplicar_estilo()
    linhas = tabela.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(linhas) + 1.4))
    ax.axvspan(*janela, color=NEUTRO, zorder=0, lw=0)
    for i, linha in linhas.iterrows():
        inicio = max(linha["inicio"], inicio_eixo)
        fim = linha["fim"] + pd.offsets.YearEnd(0) if linha["frequencia"] == "YS" else linha["fim"]
        ax.barh(i, fim - inicio, left=inicio, height=0.5, color=SERIE_1, zorder=2)
        if linha["inicio"] < inicio_eixo:
            ax.text(inicio_eixo, i, "◂ ", ha="right", va="center", color=TEXTO_SECUNDARIO)
    ax.set_yticks(range(len(linhas)), linhas["linha_figura"])
    ax.invert_yaxis()
    ax.set_ylim(len(linhas) - 0.4, -0.6)
    ax.set_xlim(inicio_eixo - pd.DateOffset(years=1), linhas["fim"].max() + pd.DateOffset(years=1))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(titulo)
    return fig


# Escala divergente do SPI em classes de McKee: seco (vermelho) <- normal (cinza) -> úmido (azul).
# Braços validados como rampas ordinais (monotonia de luminosidade e contraste do extremo claro).
CLASSES_SPI = [-float("inf"), -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, float("inf")]
CORES_SPI = ["#a3292b", "#e34948", "#ef8f87", NEUTRO, "#86b6ef", "#2a78d6", "#104281"]
ROTULOS_SPI = [
    "≤ −2 extremamente seco", "−2 a −1,5 muito seco", "−1,5 a −1 moderadamente seco",
    "−1 a 1 normal", "1 a 1,5 moderadamente úmido", "1,5 a 2 muito úmido", "≥ 2 extremamente úmido",
]  # fmt: skip


def mapas_spi(malha, valores: pd.DataFrame, paineis: list[tuple[str, pd.Timestamp]],
              coluna: str, titulo: str) -> plt.Figure:  # fmt: skip
    """Mapas em painéis pequenos do SPI por microrregião, em classes de McKee.

    `malha` é o GeoDataFrame das microrregiões (code_micro); `valores` tem codigo, mes e
    `coluna`; `paineis` é uma lista de (título do painel, mês).
    """
    from matplotlib.patches import Patch

    aplicar_estilo()
    fig, eixos = plt.subplots(1, len(paineis), figsize=(3.2 * len(paineis), 4.2))
    malha = malha.assign(codigo=malha["code_micro"].astype(str))
    for ax, (rotulo, mes) in zip(eixos, paineis, strict=True):
        dados = malha.merge(valores[valores["mes"] == mes], on="codigo", how="left")
        classe = pd.cut(dados[coluna], CLASSES_SPI, labels=False)
        cores = [CORES_SPI[int(c)] if pd.notna(c) else SUPERFICIE for c in classe]
        dados.plot(ax=ax, color=cores, edgecolor=SUPERFICIE, linewidth=0.15)
        ax.set_title(f"{rotulo}\n{mes:%Y-%m}", fontsize=9, loc="center")
        ax.set_axis_off()
    legenda = [
        Patch(facecolor=c, edgecolor=GRADE, label=r)
        for c, r in zip(CORES_SPI, ROTULOS_SPI, strict=True)
    ]
    fig.legend(handles=legenda, loc="lower center", ncol=4, frameon=False, fontsize=8,
               labelcolor=TEXTO_SECUNDARIO)  # fmt: skip
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXTO)
    fig.subplots_adjust(bottom=0.17, top=0.82, wspace=0.05)
    return fig


def series_spi_eventos(series: pd.DataFrame, eventos: list[dict], coluna: str,
                       inicio_eixo: pd.Timestamp, titulo: str) -> plt.Figure:  # fmt: skip
    """Uma série de SPI por evento (painéis), com a janela documentada sombreada e a linha −1."""
    aplicar_estilo()
    fig, eixos = plt.subplots(len(eventos), 1, figsize=(9, 1.9 * len(eventos)), sharex=True)
    for ax, evento in zip(eixos, eventos, strict=True):
        serie = series[(series["evento"] == evento["evento"]) & (series["mes"] >= inicio_eixo)]
        ax.axvspan(pd.Timestamp(evento["inicio"]), pd.Timestamp(evento["fim"]), color=NEUTRO, lw=0)
        ax.axhline(0, color=GRADE, lw=1)
        ax.axhline(-1, color=TEXTO_SECUNDARIO, lw=0.8, ls=(0, (4, 3)))
        ax.plot(serie["mes"], serie[coluna], color=SERIE_1, lw=1.4)
        ax.set_ylim(-3.2, 3.2)
        ax.set_yticks([-3, -1, 0, 1, 3])
        ax.set_title(f"{evento['evento']} — UFs: {', '.join(evento['ufs'])}", fontsize=9)
        ax.grid(axis="x", visible=False)
    eixos[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    eixos[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXTO)
    fig.text(0.01, 0.005, "Faixa cinza: janela da seca documentada. Linha tracejada: SPI = −1.",
             fontsize=8, color=TEXTO_SECUNDARIO)  # fmt: skip
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return fig


NOMES_CULTURAS = {
    "soja": "Soja", "milho": "Milho", "milho_1a_safra": "Milho 1ª safra",
    "milho_2a_safra": "Milho 2ª safra", "arroz": "Arroz", "feijao": "Feijão",
}  # fmt: skip

# Rampa sequencial azul (magnitude): classes de participação na produção nacional (%).
CLASSES_PESO = [-0.000001, 0.0, 0.1, 0.5, 1.0, 2.0, 5.0, float("inf")]
CORES_PESO = [NEUTRO, "#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
ROTULOS_PESO = ["0", "até 0,1%", "0,1 a 0,5%", "0,5 a 1%", "1 a 2%", "2 a 5%", "acima de 5%"]


def mapas_pesos(
    malha, paineis: list[tuple[str, pd.DataFrame]], contornos: dict, titulo: str
) -> plt.Figure:
    """Participação (%) de cada microrregião na produção nacional, em classes, com contornos.

    `paineis` é uma lista de (título, DataFrame com codigo e peso); `contornos` mapeia um nome a
    uma geometria (contornos de regiões, desenhados com linha de tinta).
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    aplicar_estilo()
    fig, eixos = plt.subplots(1, len(paineis), figsize=(3.4 * len(paineis), 4.4))
    malha = malha.assign(codigo=malha["code_micro"].astype(str))
    estilos = [(0, ()), (0, (3, 2))]
    for ax, (rotulo, pesos) in zip(eixos, paineis, strict=True):
        dados = malha.merge(pesos, on="codigo", how="left")
        classe = pd.cut(100 * dados["peso"].fillna(0), CLASSES_PESO, labels=False)
        dados.plot(
            ax=ax, color=[CORES_PESO[int(c)] for c in classe], edgecolor=SUPERFICIE, linewidth=0.15
        )
        for geometria, estilo in zip(contornos.values(), estilos, strict=True):
            geometria.boundary.plot(ax=ax, color=TEXTO, linewidth=0.8, linestyle=estilo)
        ax.set_title(rotulo, fontsize=9, loc="center")
        ax.set_axis_off()
    legenda = [
        Patch(facecolor=c, edgecolor=GRADE, label=r)
        for c, r in zip(CORES_PESO, ROTULOS_PESO, strict=True)
    ]
    legenda += [
        Line2D([], [], color=TEXTO, lw=0.8, ls=e, label=n)
        for n, e in zip(contornos, estilos, strict=True)
    ]
    fig.legend(
        handles=legenda,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8,
        labelcolor=TEXTO_SECUNDARIO,
    )
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXTO)
    fig.subplots_adjust(bottom=0.17, top=0.85, wspace=0.05)
    return fig


def numero(valor: float, casas: int = 2) -> str:
    """Número com vírgula decimal (texto das figuras em português)."""
    return f"{valor:.{casas}f}".replace(".", ",")


def mapa_cinturoes(malha, sedes, raios_km: list[float], titulo: str,
                   rotulos_a_esquerda: tuple[str, ...] = ()) -> plt.Figure:  # fmt: skip
    """Sedes das áreas do IPCA com círculos de `raios_km` (linha cheia e tracejada).

    Os nomes em `rotulos_a_esquerda` ficam à esquerda do ponto, para não colidir com vizinhos.
    """
    import geopandas as gpd

    aplicar_estilo()
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    malha.plot(ax=ax, color=NEUTRO, edgecolor=SUPERFICIE, linewidth=0.2)
    pontos = gpd.GeoSeries(gpd.points_from_xy(sedes["lon"], sedes["lat"]), crs=4326)
    for raio, estilo in zip(raios_km, [(0, ()), (0, (3, 2))], strict=True):
        circulos = pontos.to_crs(5880).buffer(raio * 1000).to_crs(malha.crs)
        circulos.boundary.plot(ax=ax, color=SERIE_1, linewidth=0.9, linestyle=estilo)
    pontos.to_crs(malha.crs).plot(ax=ax, color=TEXTO, markersize=6)
    for sede in sedes.itertuples(index=False):
        esquerda = sede.nome in rotulos_a_esquerda
        ax.annotate(sede.nome, (sede.lon, sede.lat), xytext=(-4, 3) if esquerda else (4, 3),
                    textcoords="offset points", ha="right" if esquerda else "left",
                    fontsize=6.5, color=TEXTO_SECUNDARIO)  # fmt: skip
    ax.set_axis_off()
    ax.set_title(titulo, fontsize=11)
    return fig


def dispersao_cwi_rendimento(tabela: pd.DataFrame, resumo: pd.DataFrame, titulo: str) -> plt.Figure:
    """Um painel por cultura: CWI (eixo x) contra a anomalia de rendimento (eixo y), por safra.

    As safras de quebra conhecida recebem rótulo com o ano; os demais pontos ficam sem rótulo.
    """
    aplicar_estilo()
    culturas = list(resumo["cultura"])
    fig, eixos = plt.subplots(1, len(culturas), figsize=(4.6 * len(culturas), 4.0))
    for ax, cultura in zip(eixos, culturas, strict=True):
        dados = tabela[tabela["cultura"] == cultura]
        linha = resumo.set_index("cultura").loc[cultura]
        ax.axhline(0, color=GRADE, lw=1)
        ax.axvline(0, color=GRADE, lw=1)
        ax.scatter(dados["cwi"], dados["anomalia_rendimento_pct"], s=28, color=SERIE_1,
                   edgecolor=SUPERFICIE, linewidth=1.5, zorder=3)  # fmt: skip
        for safra in dados[dados["ano_de_quebra"]].itertuples(index=False):
            ponto = (safra.cwi, safra.anomalia_rendimento_pct)
            ax.annotate(str(safra.safra), ponto, xytext=(5, -3), textcoords="offset points",
                        fontsize=8, color=TEXTO)  # fmt: skip
        nome = NOMES_CULTURAS.get(cultura, cultura)
        r, p = numero(linha["correlacao"]), numero(linha["p_valor"], 3)
        ax.set_title(f"{nome}: r = {r} (p = {p}, n = {linha['n']})", fontsize=9)
        ax.set_xlabel("CWI (SPI-3 ponderado pela produção, último mês crítico)")
        ax.set_ylabel("Rendimento: desvio da tendência (%)")
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXTO)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


NOMES_GRUPOS = {
    "alimentacao_domicilio": "Alimentação no domicílio",
    "alimentacao_bebidas": "Alimentação e bebidas",
    "hortalicas_verduras": "Hortaliças e verduras",
    "tuberculos_raizes_legumes": "Tubérculos, raízes e legumes",
    "frutas": "Frutas",
    "cereais_leguminosas": "Cereais e leguminosas",
    "carnes": "Carnes",
    "aves_ovos": "Aves e ovos",
    "leite_derivados": "Leite e derivados",
    "indice_geral": "Índice geral",
    "oleo_diesel": "Diesel",
}


def perfil_sazonal(tabela: pd.DataFrame, titulo: str) -> plt.Figure:
    """Barras do fator sazonal médio (pontos de log x 100) por mês do ano, um painel por grupo."""
    aplicar_estilo()
    grupos = list(dict.fromkeys(tabela["grupo"]))
    colunas = 3
    linhas = int(np.ceil(len(grupos) / colunas))
    fig, eixos = plt.subplots(linhas, colunas, figsize=(9, 2.1 * linhas), sharex=True)
    for ax, grupo in zip(eixos.ravel(), grupos, strict=False):
        dados = tabela[tabela["grupo"] == grupo]
        ax.bar(dados["mes_do_ano"], dados["fator_sazonal"], color=SERIE_1, width=0.7)
        ax.axhline(0, color=GRADE, lw=1)
        ax.set_title(NOMES_GRUPOS.get(grupo, grupo), fontsize=9)
        ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
        ax.grid(axis="x", visible=False)
    for ax in eixos.ravel()[len(grupos) :]:
        ax.set_visible(False)
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXTO)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def ccf_paineis(tabela: pd.DataFrame, hipoteses: dict, titulo: str) -> plt.Figure:
    """Correlação cruzada pré-branqueada por par (painéis), com bandas e a janela da hipótese.

    Barras: correlação entre o clima em t−k e a inflação em t. Linha tracejada: banda pontual
    de 95%; pontilhada: banda de Bonferroni. Faixa cinza: defasagens da hipótese.
    """
    aplicar_estilo()
    pares = list(dict.fromkeys(zip(tabela["grupo"], tabela["clima"], strict=True)))
    colunas = 3
    linhas = int(np.ceil(len(pares) / colunas))
    fig, eixos = plt.subplots(linhas, colunas, figsize=(10, 2.2 * linhas), sharex=True, sharey=True)
    for ax, (grupo, clima) in zip(eixos.ravel(), pares, strict=False):
        dados = tabela[(tabela["grupo"] == grupo) & (tabela["clima"] == clima)]
        if grupo in hipoteses:
            inicio, fim = hipoteses[grupo]
            ax.axvspan(inicio - 0.5, fim + 0.5, color=NEUTRO, lw=0)
        ax.bar(dados["defasagem"], dados["r"], color=SERIE_1, width=0.6)
        for banda, estilo in [("banda_pontual", (0, (4, 3))), ("banda_bonferroni", (0, (1, 2)))]:
            valor = dados[banda].iloc[0]
            for sinal in (1, -1):
                ax.axhline(sinal * valor, color=TEXTO_SECUNDARIO, lw=0.8, ls=estilo)
        ax.axhline(0, color=GRADE, lw=1)
        ax.set_title(f"{NOMES_GRUPOS.get(grupo, grupo)}\n× {clima}", fontsize=8)
        ax.grid(axis="x", visible=False)
    for ax in eixos.ravel()[len(pares) :]:
        ax.set_visible(False)
    for ax in eixos[-1]:
        ax.set_xlabel("defasagem k (meses)")
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXTO)
    nota = (
        "Barras: correlação entre o clima em t−k e a inflação dessazonalizada em t, após "
        "pré-branqueamento. Tracejado: banda pontual de 95%; pontilhado: Bonferroni "
        "(19 defasagens). Faixa cinza: janela da hipótese."
    )
    fig.text(0.01, 0.005, nota, fontsize=7.5, color=TEXTO_SECUNDARIO, wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return fig


def wavelet_paineis(tabela: pd.DataFrame, titulo: str) -> plt.Figure:
    """Coerência wavelet por par: cor = coerência (0 a 1, rampa azul), contorno = significativa
    a 95% (Monte Carlo), área clareada = fora do cone de influência (efeito de borda)."""
    from matplotlib.colors import LinearSegmentedColormap

    aplicar_estilo()
    rampa = LinearSegmentedColormap.from_list(
        "azul", ["#f0efec", "#b7d3f6", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]
    )
    pares = list(dict.fromkeys(zip(tabela["grupo"], tabela["clima"], strict=True)))
    fig, eixos = plt.subplots(len(pares), 1, figsize=(9, 2.4 * len(pares)), sharex=True)
    for ax, (grupo, clima) in zip(np.atleast_1d(eixos), pares, strict=True):
        dados = tabela[(tabela["grupo"] == grupo) & (tabela["clima"] == clima)]
        grade = dados.pivot(index="periodo", columns="mes", values="coerencia")
        sig = dados.pivot(index="periodo", columns="mes", values="significativa").astype(float)
        coi = dados.pivot(index="periodo", columns="mes", values="dentro_coi")
        x = mdates.date2num(grade.columns)
        y = np.log2(grade.index.to_numpy())
        malha = ax.pcolormesh(x, y, grade.to_numpy(), cmap=rampa, vmin=0, vmax=1, shading="auto")
        ax.contour(x, y, sig.to_numpy(), levels=[0.5], colors=TEXTO, linewidths=0.8)
        ax.contourf(x, y, (~coi).astype(float).to_numpy(), levels=[0.5, 1.5], colors=[SUPERFICIE],
                    alpha=0.65)  # fmt: skip
        ticks = [4, 8, 16, 32, 64]
        ax.set_yticks(np.log2(ticks), [str(t) for t in ticks])
        ax.set_ylim(y.min(), np.log2(64))
        ax.set_ylabel("período (meses)")
        ax.set_title(f"{NOMES_GRUPOS.get(grupo, grupo)} × {clima}", fontsize=9)
        ax.grid(False)
    ultimo = np.atleast_1d(eixos)[-1]
    ultimo.xaxis_date()
    ultimo.xaxis.set_major_locator(mdates.YearLocator(2))
    ultimo.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.colorbar(malha, ax=eixos, shrink=0.6, label="coerência")
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXTO)
    return fig


# Paleta categórica de referência (ordem fixa; os 4 primeiros passam nas checagens de pares
# adjacentes; aqua e amarelo exigem rótulo direto ou tabela por contraste < 3:1).
CATEGORICAS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
ESTILOS_LINHA = ["-", "--", "-.", ":"]
BANDA_90 = "#cde2fb"
BANDA_68 = "#86b6ef"


def irf_paineis(irf: pd.DataFrame, choques: list[str], rotulos: dict[str, str], titulo: str,
                nota: str) -> plt.Figure:  # fmt: skip
    """Resposta acumulada por choque (painéis): linha = estimativa, faixas = 68% e 90%."""
    aplicar_estilo()
    fig, eixos = plt.subplots(1, len(choques), figsize=(3.6 * len(choques), 3.6), sharey=True)
    for ax, choque in zip(np.atleast_1d(eixos), choques, strict=True):
        dados = irf[irf["choque"] == choque].sort_values("horizonte")
        h = dados["horizonte"]
        ax.fill_between(h, dados["li_90"], dados["ls_90"], color=BANDA_90, lw=0, label="90%")
        ax.fill_between(h, dados["li_68"], dados["ls_68"], color=BANDA_68, lw=0, label="68%")
        ax.plot(h, dados["coef"], color=SERIE_1, lw=2, marker="o", ms=3)
        ax.axhline(0, color=TEXTO_SECUNDARIO, lw=0.8)
        ax.set_title(rotulos.get(choque, choque), fontsize=9)
        ax.set_xlabel("meses após o choque (h)")
        ax.set_xticks(range(0, int(h.max()) + 1, 3))
    np.atleast_1d(eixos)[0].set_ylabel("variação acumulada do preço (p.p.)")
    np.atleast_1d(eixos)[0].legend(loc="upper left", frameon=False, fontsize=8, title="bandas",
                                   title_fontsize=8)  # fmt: skip
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXTO)
    fig.text(0.01, 0.005, nota, fontsize=7.5, color=TEXTO_SECUNDARIO, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    return fig


def irf_comparacao(irf: pd.DataFrame, choques: list[str], especificacoes: list[str],
                   rotulos: dict[str, str], nomes_especificacoes: dict[str, str],
                   titulo: str) -> plt.Figure:  # fmt: skip
    """Estimativas pontuais de várias especificações por choque, com rótulo direto no fim."""
    aplicar_estilo()
    fig, eixos = plt.subplots(1, len(choques), figsize=(3.9 * len(choques), 3.8), sharey=True)
    for ax, choque in zip(np.atleast_1d(eixos), choques, strict=True):
        for i, espec in enumerate(especificacoes):
            dados = irf[(irf["choque"] == choque) & (irf["especificacao"] == espec)]
            dados = dados.sort_values("horizonte")
            nome = nomes_especificacoes.get(espec, espec)
            ax.plot(dados["horizonte"], dados["coef"], color=CATEGORICAS[i], ls=ESTILOS_LINHA[i],
                    lw=2 if i == 0 else 1.4, label=nome)  # fmt: skip
        ax.axhline(0, color=TEXTO_SECUNDARIO, lw=0.8)
        ax.set_title(rotulos.get(choque, choque), fontsize=9)
        ax.set_xlabel("meses após o choque (h)")
        ax.set_xticks(range(0, 19, 3))
    np.atleast_1d(eixos)[0].set_ylabel("variação acumulada do preço (p.p.)")
    alcas, nomes = np.atleast_1d(eixos)[0].get_legend_handles_labels()
    fig.legend(alcas, nomes, loc="lower center", ncol=len(especificacoes), frameon=False,
               fontsize=8, labelcolor=TEXTO_SECUNDARIO)  # fmt: skip
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXTO)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    return fig


def _virgula(ax, eixo: str = "y") -> None:
    """Rótulos numéricos do eixo com vírgula decimal."""
    formatador = mticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))
    (ax.yaxis if eixo == "y" else ax.xaxis).set_major_formatter(formatador)


JANELA_HIPOTESE = "#e4e3df"


def irf_grade(irf: pd.DataFrame, paineis: list[tuple[str, str]], titulos: dict, titulo: str,
              nota: str, sufixo: str = "_dp", ncols: int = 4,
              janelas: dict[str, tuple[int, int]] | None = None,
              comparacao: pd.DataFrame | None = None, nome_comparacao: str = "",
              rotulo_y: str = "variação acumulada do preço (p.p.)") -> plt.Figure:  # fmt: skip
    """Grade de respostas, um painel por (dependente, choque), cada um com seu eixo y.

    `irf` (uma especificação) tem dependente, choque, horizonte e coef/li_/ls_ com `sufixo`.
    `janelas` sombreia a janela de defasagem da hipótese de cada dependente; `comparacao`
    (mesmo formato) entra como estimativa pontual tracejada.
    """
    aplicar_estilo()
    linhas = int(np.ceil(len(paineis) / ncols))
    altura = 2.8 * linhas + 1.1
    fig, eixos = plt.subplots(linhas, ncols, figsize=(3.4 * ncols, altura), squeeze=False)
    for i, (ax, (dep, choque)) in enumerate(zip(eixos.flat, paineis, strict=False)):
        dados = irf[(irf["dependente"] == dep) & (irf["choque"] == choque)]
        dados = dados.sort_values("horizonte")
        h = dados["horizonte"]
        if janelas and dep in janelas:
            a, b = janelas[dep]
            ax.axvspan(a - 0.5, b + 0.5, color=JANELA_HIPOTESE, lw=0, zorder=0,
                       label="janela da hipótese")  # fmt: skip
        ax.fill_between(h, dados[f"li_90{sufixo}"], dados[f"ls_90{sufixo}"], color=BANDA_90,
                        lw=0, label="banda de 90%")  # fmt: skip
        ax.fill_between(h, dados[f"li_68{sufixo}"], dados[f"ls_68{sufixo}"], color=BANDA_68,
                        lw=0, label="banda de 68%")  # fmt: skip
        ax.plot(h, dados[f"coef{sufixo}"], color=SERIE_1, lw=2, marker="o", ms=2.5,
                label="estimativa")  # fmt: skip
        if comparacao is not None:
            comp = comparacao[(comparacao["dependente"] == dep) & (comparacao["choque"] == choque)]
            comp = comp.sort_values("horizonte")
            ax.plot(comp["horizonte"], comp[f"coef{sufixo}"], color=TEXTO_SECUNDARIO, lw=1.2,
                    ls="--", label=nome_comparacao)  # fmt: skip
        ax.axhline(0, color=TEXTO_SECUNDARIO, lw=0.8)
        ax.set_title(titulos.get((dep, choque), f"{dep} × {choque}"), fontsize=8.5)
        ax.set_xticks(range(0, 19, 3))
        ax.set_xlim(-0.6, 18.6)
        _virgula(ax)
        if i % ncols == 0:
            ax.set_ylabel(rotulo_y, fontsize=8)
        if i >= len(paineis) - ncols:
            ax.set_xlabel("meses após o choque (h)", fontsize=8)
    for ax in eixos.flat[len(paineis) :]:
        ax.set_visible(False)
    alcas, nomes = eixos.flat[0].get_legend_handles_labels()
    for ax in eixos.flat[1 : len(paineis)]:
        for alca, nome in zip(*ax.get_legend_handles_labels(), strict=True):
            if nome not in nomes:
                alcas.append(alca)
                nomes.append(nome)
    # O tight_layout já reserva o espaço do título; a legenda ocupa ~0,3 polegada abaixo dele.
    fig.legend(alcas, nomes, loc="upper left", bbox_to_anchor=(0.005, 1 - 0.38 / altura),
               ncol=len(nomes), frameon=False, fontsize=8, labelcolor=TEXTO_SECUNDARIO)  # fmt: skip
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXTO)
    fig.text(0.01, 0.005, nota, fontsize=7.5, color=TEXTO_SECUNDARIO, wrap=True)
    fig.tight_layout(rect=(0, 0.45 / altura, 1, 1 - 0.3 / altura))
    return fig


# Choques extremos: laranja = seca, azul = excesso de chuva, amarelo = calor (validados no modo
# claro; o amarelo tem contraste < 3:1 e por isso vem com marcador próprio, legenda e tabela).
CORES_EXTREMOS = {"seca": "#eb6834", "excesso": "#2a78d6", "calor": "#eda100"}
MARCADORES_EXTREMOS = {"seca": "o", "excesso": "s", "calor": "^"}


def _floresta_eixo(ax, tabela: pd.DataFrame, modelos: list[str], rotulos_modelos: dict[str, str],
                   nomes_choques: dict[str, str], cores: dict[str, str],
                   marcadores: dict[str, str]) -> None:  # fmt: skip
    """Um painel de floresta: uma linha por modelo, um ponto e um traço por choque."""
    choques = list(nomes_choques)
    deslocamentos = np.linspace(-0.24, 0.24, len(choques))
    for j, choque in enumerate(choques):
        dados = tabela[tabela["choque"] == choque].set_index("modelo").reindex(modelos)
        y = np.arange(len(modelos)) + deslocamentos[j]
        ax.hlines(y, dados["li_90"], dados["ls_90"], color=cores[choque], lw=1.6)
        ax.plot(dados["coef"], y, ls="none", marker=marcadores[choque], ms=6.5,
                color=cores[choque], mec=SUPERFICIE, mew=1.2,
                label=nomes_choques[choque])  # fmt: skip
    ax.axvline(0, color=TEXTO_SECUNDARIO, lw=0.8)
    ax.set_yticks(range(len(modelos)), [rotulos_modelos[m] for m in modelos], fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(modelos) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)


def floresta_choques(tabela: pd.DataFrame, modelos: list[str], rotulos_modelos: dict[str, str],
                     nomes_choques: dict[str, str], titulo: str, nota: str,
                     rotulo_x: str) -> plt.Figure:  # fmt: skip
    """Resposta a cada tipo de choque num horizonte fixo, uma linha por modelo.

    `tabela` tem modelo, choque (chaves de `nomes_choques`), coef, li_90 e ls_90, já na escala
    a exibir. Ponto = estimativa; traço = banda de 90%.
    """
    aplicar_estilo()
    altura = 0.62 * len(modelos) + 1.9
    fig, ax = plt.subplots(figsize=(8.6, altura))
    _floresta_eixo(ax, tabela, modelos, rotulos_modelos, nomes_choques, CORES_EXTREMOS,
                   MARCADORES_EXTREMOS)  # fmt: skip
    ax.set_xlabel(rotulo_x, fontsize=8.5)
    _virgula(ax, "x")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(nomes_choques), frameon=False,
              fontsize=8.5, labelcolor=TEXTO_SECUNDARIO)  # fmt: skip
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXTO)
    fig.text(0.01, 0.005, nota, fontsize=7.5, color=TEXTO_SECUNDARIO, wrap=True)
    fig.tight_layout(rect=(0, 0.55 / altura, 1, 1))
    return fig


def floresta_grade(tabela: pd.DataFrame, paineis: list[str], titulos_paineis: dict[str, str],
                   modelos: list[str], rotulos_modelos: dict[str, str],
                   nomes_choques: dict[str, str], titulo: str, nota: str, rotulo_x: str,
                   ncols: int = 2) -> plt.Figure:  # fmt: skip
    """Florestas lado a lado, uma por painel (coluna `painel` de `tabela`), cada uma com seu
    eixo x. Choques nas cores categóricas em ordem fixa, com marcador próprio (o 3º, aqua,
    tem contraste < 3:1: a identidade vem do marcador, da legenda e da tabela)."""
    aplicar_estilo()
    choques = list(nomes_choques)
    cores = dict(zip(choques, CATEGORICAS, strict=False))
    marcadores = dict(zip(choques, ["o", "s", "^", "D"], strict=False))
    linhas = int(np.ceil(len(paineis) / ncols))
    altura_painel = 0.36 * len(modelos) + 0.9
    altura = altura_painel * linhas + 1.3
    fig, eixos = plt.subplots(linhas, ncols, figsize=(5.2 * ncols, altura), squeeze=False)
    for i, (ax, painel) in enumerate(zip(eixos.flat, paineis, strict=False)):
        dados = tabela[tabela["painel"] == painel]
        _floresta_eixo(ax, dados, modelos, rotulos_modelos, nomes_choques, cores, marcadores)
        ax.set_title(titulos_paineis.get(painel, painel), fontsize=9.5)
        if i % ncols:
            ax.set_yticklabels([])
        if i >= len(paineis) - ncols:
            ax.set_xlabel(rotulo_x, fontsize=8)
        _virgula(ax, "x")
    for ax in eixos.flat[len(paineis) :]:
        ax.set_visible(False)
    alcas, nomes = eixos.flat[0].get_legend_handles_labels()
    fig.legend(alcas, nomes, loc="upper left", bbox_to_anchor=(0.005, 1 - 0.38 / altura),
               ncol=len(nomes), frameon=False, fontsize=8.5,
               labelcolor=TEXTO_SECUNDARIO)  # fmt: skip
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXTO)
    fig.text(0.01, 0.005, nota, fontsize=7.5, color=TEXTO_SECUNDARIO, wrap=True)
    fig.tight_layout(rect=(0, 0.5 / altura, 1, 1 - 0.32 / altura))
    return fig


def previsao_grade(tabela: pd.DataFrame, paineis: list[str], titulos_paineis: dict[str, str],
                   series: dict[str, str], titulo: str, nota: str, rotulo_y: str,
                   rotulo_destaque: str, ncols: int = 4) -> plt.Figure:  # fmt: skip
    """Uma métrica de previsão por horizonte, um painel por grupo (coluna `painel`).

    `tabela` tem painel, serie (chaves de `series`), horizonte, valor e destaque (booleano).
    Séries nas cores categóricas em ordem fixa, com marcador próprio; marcador cheio = destaque
    (por exemplo, teste significativo), vazio = sem destaque.
    """
    aplicar_estilo()
    marcadores = ["o", "s", "^", "D"]
    linhas = int(np.ceil(len(paineis) / ncols))
    altura = 2.6 * linhas + 1.3
    fig, eixos = plt.subplots(linhas, ncols, figsize=(3.4 * ncols, altura), squeeze=False)
    horizontes = sorted(tabela["horizonte"].unique())
    for i, (ax, painel) in enumerate(zip(eixos.flat, paineis, strict=False)):
        for j, (serie, rotulo) in enumerate(series.items()):
            dados = tabela[(tabela["painel"] == painel) & (tabela["serie"] == serie)]
            dados = dados.sort_values("horizonte")
            cor = CATEGORICAS[j]
            ax.plot(dados["horizonte"], dados["valor"], color=cor, lw=1.6, label=rotulo)
            cheio, vazio = dados[dados["destaque"]], dados[~dados["destaque"]]
            ax.plot(cheio["horizonte"], cheio["valor"], ls="none", marker=marcadores[j], ms=7,
                    color=cor, mec=SUPERFICIE, mew=1.0)  # fmt: skip
            ax.plot(vazio["horizonte"], vazio["valor"], ls="none", marker=marcadores[j], ms=6,
                    mfc=SUPERFICIE, mec=cor, mew=1.4)  # fmt: skip
        ax.axhline(0, color=TEXTO_SECUNDARIO, lw=0.8)
        ax.set_title(titulos_paineis.get(painel, painel), fontsize=9)
        ax.set_xticks(horizontes)
        _virgula(ax)
        if i % ncols == 0:
            ax.set_ylabel(rotulo_y, fontsize=8)
        if i >= len(paineis) - ncols:
            ax.set_xlabel("horizonte da previsão (meses)", fontsize=8)
    for ax in eixos.flat[len(paineis) :]:
        ax.set_visible(False)
    alcas, nomes = eixos.flat[0].get_legend_handles_labels()
    alcas += [
        plt.Line2D([], [], ls="none", marker="o", ms=7, color=TEXTO_SECUNDARIO),
        plt.Line2D([], [], ls="none", marker="o", ms=6, mfc=SUPERFICIE, mec=TEXTO_SECUNDARIO),
    ]
    nomes += [rotulo_destaque, "não"]
    fig.legend(alcas, nomes, loc="upper left", bbox_to_anchor=(0.005, 1 - 0.38 / altura),
               ncol=len(nomes), frameon=False, fontsize=8, labelcolor=TEXTO_SECUNDARIO)  # fmt: skip
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXTO)
    fig.text(0.01, 0.005, nota, fontsize=7.5, color=TEXTO_SECUNDARIO, wrap=True)
    fig.tight_layout(rect=(0, 0.5 / altura, 1, 1 - 0.3 / altura))
    return fig


def erro_acumulado_paineis(tabela: pd.DataFrame, paineis: list[str],
                           titulos_paineis: dict[str, str], series: dict[str, str], titulo: str,
                           nota: str, rotulo_y: str,
                           faixa: tuple[pd.Timestamp, pd.Timestamp, str] | None = None,
                           ) -> plt.Figure:  # fmt: skip
    """Série acumulada ao longo do tempo (por exemplo, erro quadrático do benchmark menos o do
    modelo), um painel por caso. `tabela` tem painel, serie, mes e valor. `faixa` sombreia um
    período de referência (início, fim, rótulo)."""
    aplicar_estilo()
    fig, eixos = plt.subplots(1, len(paineis), figsize=(4.4 * len(paineis), 3.9), squeeze=False)
    for i, (ax, painel) in enumerate(zip(eixos.flat, paineis, strict=True)):
        if faixa is not None:
            ax.axvspan(faixa[0], faixa[1], color=JANELA_HIPOTESE, lw=0, zorder=0, label=faixa[2])
        for j, (serie, rotulo) in enumerate(series.items()):
            dados = tabela[(tabela["painel"] == painel) & (tabela["serie"] == serie)]
            dados = dados.sort_values("mes")
            ax.plot(dados["mes"], dados["valor"], color=CATEGORICAS[j], ls=ESTILOS_LINHA[j],
                    lw=1.8, label=rotulo)  # fmt: skip
        ax.axhline(0, color=TEXTO_SECUNDARIO, lw=0.8)
        ax.set_title(titulos_paineis.get(painel, painel), fontsize=9)
        ax.xaxis.set_major_locator(mdates.YearLocator(3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        _virgula(ax)
        if i == 0:
            ax.set_ylabel(rotulo_y, fontsize=8)
    alcas, nomes = eixos.flat[0].get_legend_handles_labels()
    fig.legend(alcas, nomes, loc="upper left", bbox_to_anchor=(0.005, 1 - 0.38 / 3.9),
               ncol=len(nomes), frameon=False, fontsize=8, labelcolor=TEXTO_SECUNDARIO)  # fmt: skip
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXTO)
    fig.text(0.01, 0.005, nota, fontsize=7.5, color=TEXTO_SECUNDARIO, wrap=True)
    fig.tight_layout(rect=(0, 0.6 / 3.9, 1, 1 - 0.3 / 3.9))
    return fig


def precos_paineis(tabela: pd.DataFrame, paineis: list[str], titulos_paineis: dict[str, str],
                   series: dict[str, str], titulo: str, nota: str, rotulo_y: str,
                   ncols: int = 3) -> plt.Figure:  # fmt: skip
    """Séries de preço (ex.: índice real, início = 100) em pequenos múltiplos, um painel por
    produto. `tabela` tem painel, serie, periodo (ano inteiro ou data) e valor; a cor segue a
    série (ordem de `series`), não o painel. Painéis sem uma das séries só mostram as demais."""
    aplicar_estilo()
    nlin = -(-len(paineis) // ncols)
    fig, eixos = plt.subplots(nlin, ncols, figsize=(4.3 * ncols, 3.2 * nlin + 0.9), squeeze=False)
    for i, (ax, painel) in enumerate(zip(eixos.flat, paineis, strict=False)):
        do_painel = tabela[tabela["painel"] == painel]
        datas = isinstance(do_painel["periodo"].iloc[0], pd.Timestamp)
        for j, (serie, rotulo) in enumerate(series.items()):
            dados = do_painel[do_painel["serie"] == serie]
            if dados.empty:
                continue
            x = pd.to_datetime(dados["periodo"]) if datas else dados["periodo"].astype(int)
            ordem = np.argsort(x.to_numpy())
            ax.plot(x.to_numpy()[ordem], dados["valor"].to_numpy()[ordem], color=CATEGORICAS[j],
                    ls=ESTILOS_LINHA[j], lw=1.8, label=rotulo)  # fmt: skip
        ax.axhline(100, color=TEXTO_SECUNDARIO, lw=0.6)
        ax.set_title(titulos_paineis.get(painel, painel), fontsize=9)
        if datas:
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        _virgula(ax)
        if i % ncols == 0:
            ax.set_ylabel(rotulo_y, fontsize=8)
    for ax in list(eixos.flat)[len(paineis) :]:
        ax.axis("off")
    alcas, nomes = {}, {}
    for ax in eixos.flat:
        for alca, nome in zip(*ax.get_legend_handles_labels(), strict=True):
            alcas.setdefault(nome, alca)
    nomes = [rotulo for rotulo in series.values() if rotulo in alcas]
    altura = 3.2 * nlin + 0.9
    fig.legend([alcas[n] for n in nomes], nomes, loc="upper left",
               bbox_to_anchor=(0.005, 1 - 0.38 / altura), ncol=len(nomes), frameon=False,
               fontsize=8, labelcolor=TEXTO_SECUNDARIO)  # fmt: skip
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXTO)
    fig.text(0.01, 0.005, nota, fontsize=7.5, color=TEXTO_SECUNDARIO, wrap=True)
    fig.tight_layout(rect=(0, 0.6 / altura, 1, 1 - 0.32 / altura))
    return fig


def multiplicadores_paineis(tabela: pd.DataFrame, longo: pd.DataFrame, paineis: list[str],
                            titulos_paineis: dict[str, str], titulo: str, nota: str,
                            rotulo_y: str) -> plt.Figure:  # fmt: skip
    """Multiplicadores dinâmicos acumulados, um painel por caso. `tabela` tem painel,
    horizonte, coef, li e ls (banda); `longo` tem painel, coef, li e ls do multiplicador de
    longo prazo, desenhado como linha tracejada com faixa clara."""
    aplicar_estilo()
    altura = 4.0
    fig, eixos = plt.subplots(1, len(paineis), figsize=(5.0 * len(paineis), altura), squeeze=False)
    for i, (ax, painel) in enumerate(zip(eixos.flat, paineis, strict=True)):
        dados = tabela[tabela["painel"] == painel].sort_values("horizonte")
        h = dados["horizonte"].astype(int)
        if painel in set(longo["painel"]):  # painéis sem relação de longo prazo ficam sem a linha
            lp = longo[longo["painel"] == painel].iloc[0]
            ax.axhspan(lp["li"], lp["ls"], color=NEUTRO, lw=0, zorder=0)
            ax.axhline(lp["coef"], color=TEXTO_SECUNDARIO, lw=1.2, ls="--",
                       label="longo prazo (faixa de 90%)")  # fmt: skip
        ax.fill_between(h, dados["li"], dados["ls"], color=BANDA_90, lw=0, label="banda de 90%")
        ax.plot(h, dados["coef"], color=SERIE_1, lw=2, marker="o", ms=2.5,
                label="multiplicador acumulado")  # fmt: skip
        ax.axhline(0, color=TEXTO_SECUNDARIO, lw=0.8)
        ax.set_title(titulos_paineis.get(painel, painel), fontsize=9.5)
        ax.set_xticks(range(0, int(h.max()) + 1, 3))
        ax.set_xlabel("meses após a alta do milho", fontsize=8)
        _virgula(ax)
        if i == 0:
            ax.set_ylabel(rotulo_y, fontsize=8)
    alcas, nomes = {}, []
    for ax in eixos.flat:
        for alca, nome in zip(*ax.get_legend_handles_labels(), strict=True):
            if nome not in alcas:
                alcas[nome] = alca
                nomes.append(nome)
    fig.legend([alcas[n] for n in nomes], nomes, loc="upper left",
               bbox_to_anchor=(0.005, 1 - 0.38 / altura), ncol=len(nomes), frameon=False,
               fontsize=8, labelcolor=TEXTO_SECUNDARIO)  # fmt: skip
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXTO)
    fig.text(0.01, 0.005, nota, fontsize=7.5, color=TEXTO_SECUNDARIO, wrap=True)
    fig.tight_layout(rect=(0, 0.62 / altura, 1, 1 - 0.32 / altura))
    return fig


# Estados da matriz de estabilidade: preenchimento claro + rótulo na legenda; o número da célula
# fica sempre em tinta de texto (a cor não é a única pista: a tabela traz o estado por extenso).
CORES_ESTADO = {
    "referência": "#ffffff",
    "mantém": "#e4e3df",
    "ganha significância": "#cde2fb",
    "perde significância": "#f8d9a8",
    "inverte": "#f4b4ae",
    "placebo passa": "#e4e3df",
    "placebo falha": "#f4b4ae",
}


def matriz_estabilidade(tabela: pd.DataFrame, linhas: list[str], colunas: list[str],
                        rotulos_linhas: dict[str, str], rotulos_colunas: dict[str, str],
                        titulo: str, nota: str) -> plt.Figure:  # fmt: skip
    """Matriz coeficiente × teste: cor = estado em relação à principal, texto = valor.

    `tabela` tem linha, coluna, estado (chave de CORES_ESTADO) e texto. Células ausentes
    ficam em branco com "—".
    """
    aplicar_estilo()
    altura = 0.3 * len(linhas) + 2.2
    largura = 3.6 + 1.05 * len(colunas)
    fig, ax = plt.subplots(figsize=(largura, altura))
    indice = tabela.set_index(["linha", "coluna"])
    for i, linha in enumerate(linhas):
        for j, coluna in enumerate(colunas):
            if (linha, coluna) in indice.index:
                celula = indice.loc[(linha, coluna)]
                cor, texto = CORES_ESTADO[celula["estado"]], celula["texto"]
                peso = "bold" if celula["estado"] == "referência" else "normal"
            else:
                cor, texto, peso = SUPERFICIE, "—", "normal"
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=cor, edgecolor=SUPERFICIE, lw=2))
            ax.text(j + 0.5, i + 0.5, texto, ha="center", va="center", fontsize=7.5, color=TEXTO,
                    fontweight=peso)  # fmt: skip
    ax.set_xlim(0, len(colunas))
    ax.set_ylim(len(linhas), 0)
    ax.set_xticks(np.arange(len(colunas)) + 0.5, [rotulos_colunas[c] for c in colunas],
                  fontsize=7.5, rotation=30, ha="left")  # fmt: skip
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(linhas)) + 0.5, [rotulos_linhas[x] for x in linhas], fontsize=7.5)
    ax.tick_params(length=0)
    ax.grid(False)
    for lado in ax.spines.values():
        lado.set_visible(False)
    presentes = [e for e in CORES_ESTADO if e in set(tabela["estado"])]
    alcas = [plt.Rectangle((0, 0), 1, 1, facecolor=CORES_ESTADO[e], edgecolor=GRADE)
             for e in presentes]  # fmt: skip
    fig.legend(alcas, presentes, loc="lower left", bbox_to_anchor=(0.005, 0.5 / altura),
               ncol=len(presentes), frameon=False, fontsize=7.5,
               labelcolor=TEXTO_SECUNDARIO)  # fmt: skip
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXTO)
    fig.text(0.01, 0.005, nota, fontsize=7.5, color=TEXTO_SECUNDARIO, wrap=True)
    fig.tight_layout(rect=(0, 0.85 / altura, 1, 1 - 0.1 / altura))
    return fig


def placebo_paineis(tabela: pd.DataFrame, paineis: list[str], titulos_paineis: dict[str, str],
                    series: dict[str, str], titulo: str, nota: str, rotulo_y: str,
                    ncols: int = 4) -> plt.Figure:  # fmt: skip
    """Respostas com banda de 90% de duas ou mais séries por painel (ex.: efeito estimado e
    placebo). `tabela` tem painel, serie, horizonte, coef, li_90 e ls_90; cores na ordem fixa
    de CATEGORICAS, com marcador próprio por série."""
    aplicar_estilo()
    nlin = -(-len(paineis) // ncols)
    altura = 2.9 * nlin + 1.2
    fig, eixos = plt.subplots(nlin, ncols, figsize=(3.6 * ncols, altura), squeeze=False)
    marcadores = ["o", "s", "^", "D"]
    for i, (ax, painel) in enumerate(zip(eixos.flat, paineis, strict=False)):
        for j, (serie, rotulo) in enumerate(series.items()):
            dados = tabela[(tabela["painel"] == painel) & (tabela["serie"] == serie)]
            dados = dados.sort_values("horizonte")
            ax.fill_between(dados["horizonte"], dados["li_90"], dados["ls_90"],
                            color=CATEGORICAS[j], alpha=0.16, lw=0)  # fmt: skip
            ax.plot(dados["horizonte"], dados["coef"], color=CATEGORICAS[j], lw=1.8,
                    marker=marcadores[j], ms=3.5, label=rotulo)  # fmt: skip
        ax.axhline(0, color=TEXTO_SECUNDARIO, lw=0.8)
        ax.set_title(titulos_paineis.get(painel, painel), fontsize=8.5)
        _virgula(ax)
        if i % ncols == 0:
            ax.set_ylabel(rotulo_y, fontsize=8)
        if i >= len(paineis) - ncols:
            ax.set_xlabel("meses após t (h)", fontsize=8)
    for ax in list(eixos.flat)[len(paineis) :]:
        ax.set_visible(False)
    alcas, nomes = eixos.flat[0].get_legend_handles_labels()
    fig.legend(alcas, nomes, loc="upper left", bbox_to_anchor=(0.005, 1 - 0.38 / altura),
               ncol=len(nomes), frameon=False, fontsize=8, labelcolor=TEXTO_SECUNDARIO)  # fmt: skip
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXTO)
    fig.text(0.01, 0.005, nota, fontsize=7.5, color=TEXTO_SECUNDARIO, wrap=True)
    fig.tight_layout(rect=(0, 0.5 / altura, 1, 1 - 0.32 / altura))
    return fig
