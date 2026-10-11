"""INMET (fonte complementar): estações automáticas perto das áreas do IPCA.

A rota de dados da API exige token; os dados vêm dos arquivos anuais do portal (um zip por ano,
um CSV horário por estação). As coordenadas vêm do cabeçalho de cada CSV, o que inclui estações
já desativadas (ausentes do catálogo atual da API). Só as estações a até `max_km` de algum
ponto de interesse são guardadas no cache.
"""

from __future__ import annotations

import hashlib
import io
import logging
import unicodedata
import zipfile
from typing import Any

import numpy as np
import pandas as pd

from clima_inflacao.config import caminho_projeto
from clima_inflacao.ingestao.cache import ler_ou_baixar
from clima_inflacao.ingestao.cliente_http import obter

log = logging.getLogger(__name__)

RAIO_TERRA_KM = 6371.0


def _normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sem_acento.upper().strip()


def distancia_km(lat1, lon1, lat2, lon2):
    """Distância em km pela fórmula de haversine (aceita escalares ou arrays)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * RAIO_TERRA_KM * np.arcsin(np.sqrt(a))


def ler_metadados(texto: str) -> dict[str, Any]:
    """Código, nome, UF e coordenadas do cabeçalho de um CSV do INMET (ex.: "LATITUDE:;-15,78").

    As chaves são casadas por prefixo: os arquivos de 2019 trazem os acentos como "?"
    ("ESTAC?O:", "REGI?O:").
    """
    prefixos = {
        "codigo": "CODIGO",
        "estacao": "ESTA",
        "uf": "UF",
        "lat_estacao": "LATITUDE",
        "lon_estacao": "LONGITUDE",
    }
    meta: dict[str, Any] = {}
    for linha in texto.splitlines()[:10]:
        chave, _, valor = linha.partition(";")
        chave = _normalizar(chave)
        for nome, prefixo in prefixos.items():
            if nome not in meta and chave.startswith(prefixo):
                meta[nome] = valor.strip()
    faltam = set(prefixos) - set(meta)
    if faltam:
        raise ValueError(f"cabeçalho do INMET sem {sorted(faltam)}")
    for nome in ["lat_estacao", "lon_estacao"]:
        meta[nome] = float(meta[nome].replace(",", "."))
    return meta


def ler_csv(texto: str) -> pd.DataFrame:
    """Lê um CSV horário do INMET nos dois formatos já publicados.

    Até ~2018: "DATA (YYYY-MM-DD);HORA (UTC)" com "2001-01-01;00:00" e ausente -9999. Depois:
    "Data;Hora UTC" com "2024/01/01;0000 UTC" e ausente como campo vazio. Decimal com vírgula.
    Devolve datahora_utc, precipitacao_mm e temperatura_c (bulbo seco).
    """
    linhas = texto.splitlines()
    inicio = next(
        i
        for i, linha in enumerate(linhas)
        if _normalizar(linha).startswith("DATA") and "HORA" in _normalizar(linha)
    )
    tabela = pd.read_csv(io.StringIO("\n".join(linhas[inicio:])), sep=";", dtype=str)
    colunas = {_normalizar(c): c for c in tabela.columns}
    precip = next(c for n, c in colunas.items() if n.startswith("PRECIPITA") and "TOTAL" in n)
    temp = next(c for n, c in colunas.items() if "BULBO SECO" in n)
    data = tabela.iloc[:, 0].str.strip().str.replace("/", "-", regex=False)
    hora = tabela.iloc[:, 1].str.extract(r"(\d{2})", expand=False)
    datahora = pd.to_datetime(data + " " + hora, format="%Y-%m-%d %H")

    def numero(coluna: str) -> pd.Series:
        valores = pd.to_numeric(tabela[coluna].str.strip().str.replace(",", ".", regex=False))
        return valores.mask(valores <= -9999)

    return pd.DataFrame(
        {
            "datahora_utc": datahora,
            "precipitacao_mm": numero(precip),
            "temperatura_c": numero(temp),
        }
    )


def _assinatura(pontos: pd.DataFrame, max_km: float) -> str:
    texto = (
        ";".join(f"{p.ponto_id}:{p.lat:.3f}:{p.lon:.3f}" for p in pontos.itertuples())
        + f"|{max_km}"
    )
    return hashlib.md5(texto.encode()).hexdigest()[:8]


def baixar_ano(
    ano: int, pontos: pd.DataFrame, max_km: float, config: dict[str, Any], *, refresh: bool = False
) -> pd.DataFrame:
    """Dados horários das estações a até `max_km` de algum ponto (`ponto_id`, `lat`, `lon`).

    O cache é por ano e por conjunto de pontos e raio (assinatura no nome do arquivo).
    """
    url = config["clima"]["inmet"]["url_historico"].format(ano=ano)
    nome = f"inmet_{ano}_{_assinatura(pontos, max_km)}.parquet"
    colunas = ["codigo", "estacao", "uf", "lat_estacao", "lon_estacao", "datahora_utc",
               "precipitacao_mm", "temperatura_c"]  # fmt: skip

    def baixar() -> pd.DataFrame:
        arquivo = zipfile.ZipFile(io.BytesIO(obter(url, http=config["http"]).content))
        partes = []
        for membro in arquivo.namelist():
            if not membro.upper().endswith(".CSV"):
                continue
            texto = arquivo.read(membro).decode("latin-1")
            meta = ler_metadados(texto)
            distancias = distancia_km(
                meta["lat_estacao"], meta["lon_estacao"], pontos["lat"], pontos["lon"]
            )
            if np.min(distancias) <= max_km:
                partes.append(ler_csv(texto).assign(**meta))
        if not partes:
            return pd.DataFrame(columns=colunas)
        return pd.concat(partes, ignore_index=True)[colunas]

    return ler_ou_baixar(
        caminho_projeto("dados_brutos", config) / "inmet" / nome, baixar, refresh=refresh
    )


def mensal(horario: pd.DataFrame, fuso_horas: int, min_pct_horas: float) -> pd.DataFrame:
    """Chuva (soma) e temperatura (média) mensais, na hora local `fuso_horas`.

    O mês só recebe valor se a fração de horas com dado for >= `min_pct_horas`; a fração fica
    nas colunas pct_horas_*. Horas repetidas da mesma estação (a mesma estação em dois arquivos
    do ano) são contadas uma vez só, com aviso no log; sem isso, a chuva seria somada em dobro.
    """
    repetidas = horario.duplicated(["codigo", "datahora_utc"])
    if repetidas.any():
        por_estacao = horario.loc[repetidas, "codigo"].value_counts().to_dict()
        log.warning("INMET: %d horas repetidas descartadas %s", int(repetidas.sum()), por_estacao)
        horario = horario[~repetidas]
    local = horario["datahora_utc"] + pd.Timedelta(hours=fuso_horas)
    mes = local.dt.to_period("M").dt.to_timestamp().rename("mes")
    grupos = horario.groupby([horario["codigo"], mes])
    resumo = grupos.agg(
        precipitacao_mm=("precipitacao_mm", "sum"),
        horas_precip=("precipitacao_mm", "count"),
        temperatura_c=("temperatura_c", "mean"),
        horas_temp=("temperatura_c", "count"),
    ).reset_index()
    horas_mes = resumo["mes"].dt.days_in_month * 24
    for variavel, horas in [("precipitacao_mm", "horas_precip"), ("temperatura_c", "horas_temp")]:
        pct = 100 * resumo[horas] / horas_mes
        resumo[f"pct_{horas}"] = pct.round(1)
        resumo[variavel] = resumo[variavel].where(pct >= min_pct_horas)
    return resumo.drop(columns=["horas_precip", "horas_temp"])


def escolher_estacoes(
    inmet_mensal: pd.DataFrame, estacoes: pd.DataFrame, pontos: pd.DataFrame, max_km: float
) -> pd.DataFrame:
    """Para cada ponto, a estação a até `max_km` com mais meses válidos de chuva.

    Empates são resolvidos pela menor distância. Pontos sem estação no raio ficam de fora.
    """
    meses = inmet_mensal.groupby("codigo")["precipitacao_mm"].count().rename("meses_validos")
    estacoes = estacoes.drop_duplicates("codigo").merge(meses, on="codigo", how="left")
    linhas = []
    for ponto in pontos.itertuples(index=False):
        dist = distancia_km(ponto.lat, ponto.lon, estacoes["lat_estacao"], estacoes["lon_estacao"])
        candidatas = estacoes.assign(distancia_km=dist.round(1))
        candidatas = candidatas[candidatas["distancia_km"] <= max_km]
        if candidatas.empty:
            log.warning("%s: nenhuma estação do INMET a menos de %.0f km", ponto.ponto_id, max_km)
            continue
        melhor = candidatas.sort_values(["meses_validos", "distancia_km"], ascending=[False, True])
        linhas.append(
            {
                "ponto_id": ponto.ponto_id,
                "n_candidatas": len(candidatas),
                **melhor.iloc[0].to_dict(),
            }
        )
    return pd.DataFrame(linhas)


def validar_contra_nasa(
    inmet_mensal: pd.DataFrame, clima_diario: pd.DataFrame, estacoes: pd.DataFrame
) -> pd.DataFrame:
    """Compara, por estação, a chuva e a temperatura mensais do INMET com a célula do NASA POWER.

    O mês do NASA POWER só entra se todos os dias tiverem dado. Devolve nº de meses comparados,
    correlação, correlação das anomalias (sem a média de cada mês do calendário, a métrica que
    importa para o SPI) e viés (NASA − INMET; chuva em % da média do INMET, temperatura em °C).
    """

    def anomalia(serie: pd.Series, meses: pd.Series) -> pd.Series:
        return serie - serie.groupby(meses.dt.month).transform("mean")

    dia = clima_diario.assign(mes=clima_diario["data"].dt.to_period("M").dt.to_timestamp())
    nasa = dia.groupby(["celula_id", "mes"]).agg(
        chuva_nasa=("precipitacao_mm", "sum"),
        dias_chuva=("precipitacao_mm", "count"),
        temp_nasa=("temp_media_c", "mean"),
        dias=("data", "count"),
    )
    nasa = nasa[nasa["dias_chuva"] == nasa["dias"]].reset_index()
    pares = inmet_mensal.merge(estacoes[["codigo", "ponto_id", "celula_id"]], on="codigo")
    pares = pares.merge(nasa, on=["celula_id", "mes"], how="inner")
    linhas = []
    for (ponto_id, codigo), grupo in pares.groupby(["ponto_id", "codigo"]):
        chuva = grupo.dropna(subset=["precipitacao_mm", "chuva_nasa"])
        temp = grupo.dropna(subset=["temperatura_c", "temp_nasa"])
        linhas.append(
            {
                "ponto_id": ponto_id,
                "codigo": codigo,
                "meses_chuva": len(chuva),
                "corr_chuva": chuva["precipitacao_mm"].corr(chuva["chuva_nasa"]),
                "corr_anom_chuva": anomalia(chuva["precipitacao_mm"], chuva["mes"]).corr(
                    anomalia(chuva["chuva_nasa"], chuva["mes"])
                ),
                "vies_chuva_pct": 100
                * (chuva["chuva_nasa"].mean() / chuva["precipitacao_mm"].mean() - 1),
                "meses_temp": len(temp),
                "corr_temp": temp["temperatura_c"].corr(temp["temp_nasa"]),
                "corr_anom_temp": anomalia(temp["temperatura_c"], temp["mes"]).corr(
                    anomalia(temp["temp_nasa"], temp["mes"])
                ),
                "vies_temp_c": (temp["temp_nasa"] - temp["temperatura_c"]).mean(),
            }
        )
    return pd.DataFrame(linhas).round(3)
