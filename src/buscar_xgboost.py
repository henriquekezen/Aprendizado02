import os
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from pre_processamento import (
    criar_features_modelo_bairro_categorico,
    selecionar_bairros_frequentes,
)


COLUNAS_CORRESPONDENCIA = [
    "tipo",
    "bairro",
    "area_util",
    "area_extra",
    "quartos",
    "suites",
    "vagas",
]
MODOS_BAIRRO = ["nativo", "onehot", "target_m2"]
PESOS_CORRESPONDENCIA = [0.0, 0.15, 0.30, 0.45]


def calcular_rmspe(y_real, y_previsto):
    erros_percentuais = (y_real - y_previsto) / y_real
    return np.sqrt(np.mean(erros_percentuais**2))


def corrigir_dados(df, corrigir_alvo=False):
    df = df.copy()
    df["area_util"] = df["area_util"].astype(float)
    mascara_area = (
        (df["tipo"] == "Apartamento")
        & (df["quartos"] > 0)
        & ((df["area_util"] / df["quartos"]) > 200)
    )
    df.loc[mascara_area, "area_util"] /= 10

    if corrigir_alvo:
        df.loc[df["Id"].isin([2749, 4316]), "preco"] /= 10
        df.loc[df["Id"] == 6383, "vagas"] = 3

    return df


def preparar_features(df_treino, df_validacao, modo_bairro):
    df_treino = df_treino.reset_index(drop=True)
    df_validacao = df_validacao.reset_index(drop=True)
    bairros_mantidos = selecionar_bairros_frequentes(
        df_treino,
        minimo_imoveis=10,
    )
    treino = criar_features_modelo_bairro_categorico(
        df_treino,
        bairros_mantidos,
    )
    validacao = criar_features_modelo_bairro_categorico(
        df_validacao,
        bairros_mantidos,
    )

    if modo_bairro == "nativo":
        return treino, validacao

    if modo_bairro == "onehot":
        treino = pd.get_dummies(treino, columns=["bairro"], dtype=int)
        validacao = pd.get_dummies(validacao, columns=["bairro"], dtype=int)
        validacao = validacao.reindex(columns=treino.columns, fill_value=0)
        return treino, validacao

    if modo_bairro != "target_m2":
        raise ValueError(f"Modo de bairro desconhecido: {modo_bairro}")

    bairro_treino = treino["bairro"].astype("string")
    bairro_validacao = validacao["bairro"].astype("string")
    log_preco_m2 = np.log1p(df_treino["preco"] / df_treino["area_util"])
    tabela = pd.DataFrame(
        {
            "bairro": bairro_treino,
            "log_preco_m2": log_preco_m2,
        }
    )
    estatisticas = tabela.groupby("bairro")["log_preco_m2"].agg(
        ["mean", "count"]
    )
    media_global = log_preco_m2.mean()
    suavizacao = 20
    estatisticas["media_suavizada"] = (
        estatisticas["mean"] * estatisticas["count"]
        + media_global * suavizacao
    ) / (estatisticas["count"] + suavizacao)
    estatisticas["frequencia"] = estatisticas["count"] / len(treino)

    treino["bairro_preco_m2_log"] = bairro_treino.map(
        estatisticas["media_suavizada"]
    ).astype(float)
    validacao["bairro_preco_m2_log"] = bairro_validacao.map(
        estatisticas["media_suavizada"]
    ).fillna(media_global).astype(float)
    treino["bairro_frequencia"] = bairro_treino.map(
        estatisticas["frequencia"]
    ).astype(float)
    validacao["bairro_frequencia"] = bairro_validacao.map(
        estatisticas["frequencia"]
    ).fillna(0).astype(float)
    treino = treino.drop(columns=["bairro"])
    validacao = validacao.drop(columns=["bairro"])
    return treino, validacao


def preparar_matrizes(df_treino, df_validacao, modo_bairro):
    treino, validacao = preparar_features(
        df_treino,
        df_validacao,
        modo_bairro,
    )
    return (
        treino.drop(columns=["Id", "preco"]),
        np.log1p(treino["preco"]),
        validacao.drop(columns=["Id", "preco"], errors="ignore"),
    )


def prever_correspondencias(df_treino, df_validacao):
    tabela = (
        df_treino.groupby(COLUNAS_CORRESPONDENCIA, dropna=False)["preco"]
        .median()
        .rename("preco_correspondencia")
        .reset_index()
    )
    validacao_ordenada = df_validacao.reset_index(drop=True).reset_index(
        names="ordem_original"
    )
    return (
        validacao_ordenada.merge(
            tabela,
            on=COLUNAS_CORRESPONDENCIA,
            how="left",
        )
        .sort_values("ordem_original")["preco_correspondencia"]
        .to_numpy()
    )


def aplicar_correspondencias(previsoes, correspondencias, peso):
    resultado = previsoes.copy()
    mascara = ~np.isnan(correspondencias)
    resultado[mascara] = (
        (1 - peso) * previsoes[mascara]
        + peso * correspondencias[mascara]
    )
    return resultado


def criar_modelo(configuracao):
    parametros = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "enable_categorical": configuracao["modo_bairro"] == "nativo",
        "n_estimators": configuracao["n_estimators"],
        "learning_rate": configuracao["learning_rate"],
        "grow_policy": configuracao["grow_policy"],
        "max_depth": configuracao["max_depth"],
        "max_leaves": configuracao["max_leaves"],
        "min_child_weight": configuracao["min_child_weight"],
        "subsample": configuracao["subsample"],
        "colsample_bytree": configuracao["colsample_bytree"],
        "reg_alpha": configuracao["reg_alpha"],
        "reg_lambda": configuracao["reg_lambda"],
        "gamma": configuracao["gamma"],
        "random_state": 42,
        "n_jobs": -1,
    }
    if configuracao["modo_bairro"] == "nativo":
        parametros["max_cat_to_onehot"] = 4
        parametros["max_cat_threshold"] = configuracao["max_cat_threshold"]
    return XGBRegressor(**parametros)


def criar_configuracao(modo_bairro, **alteracoes):
    configuracao = {
        "modo_bairro": modo_bairro,
        "n_estimators": 800,
        "learning_rate": 0.03,
        "grow_policy": "depthwise",
        "max_depth": 4,
        "max_leaves": 0,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "gamma": 0.0,
        "max_cat_threshold": 64,
    }
    configuracao.update(alteracoes)
    return configuracao


def chave_configuracao(configuracao):
    return tuple(sorted(configuracao.items()))


def criar_busca_estrutura():
    configuracoes = []
    for modo in MODOS_BAIRRO:
        for profundidade in [2, 3, 4, 5, 6]:
            for peso_filho in [1, 5, 12]:
                configuracoes.append(
                    criar_configuracao(
                        modo,
                        max_depth=profundidade,
                        min_child_weight=peso_filho,
                    )
                )
        for folhas in [8, 16, 32, 64]:
            for peso_filho in [1, 5]:
                configuracoes.append(
                    criar_configuracao(
                        modo,
                        grow_policy="lossguide",
                        max_depth=0,
                        max_leaves=folhas,
                        min_child_weight=peso_filho,
                    )
                )
    return configuracoes


def criar_busca_refinada(melhores_estruturas):
    configuracoes = []
    pares_boosting = [
        (300, 0.08),
        (400, 0.06),
        (600, 0.04),
        (1200, 0.02),
        (1600, 0.015),
    ]
    amostragens = [
        (0.65, 0.8),
        (1.0, 0.8),
        (0.8, 0.65),
        (0.8, 1.0),
        (1.0, 1.0),
    ]
    regularizacoes = [
        (0.0, 0.5, 0.0),
        (0.2, 3.0, 0.0),
        (1.0, 10.0, 0.0),
        (0.05, 1.0, 0.1),
    ]

    for base in melhores_estruturas:
        for arvores, taxa in pares_boosting:
            nova = deepcopy(base)
            nova.update(n_estimators=arvores, learning_rate=taxa)
            configuracoes.append(nova)
        for linhas, colunas in amostragens:
            nova = deepcopy(base)
            nova.update(subsample=linhas, colsample_bytree=colunas)
            configuracoes.append(nova)
        for alpha, lambd, gamma in regularizacoes:
            nova = deepcopy(base)
            nova.update(reg_alpha=alpha, reg_lambda=lambd, gamma=gamma)
            configuracoes.append(nova)
        if base["modo_bairro"] == "nativo":
            for limite in [16, 32, 128]:
                nova = deepcopy(base)
                nova["max_cat_threshold"] = limite
                configuracoes.append(nova)
    return configuracoes


def avaliar_no_corte(configuracao, matrizes, y_real, correspondencias):
    x_treino, y_treino_log, x_validacao = matrizes
    modelo = criar_modelo(configuracao)
    modelo.fit(x_treino, y_treino_log)
    previsoes = np.expm1(modelo.predict(x_validacao))
    return {
        peso: calcular_rmspe(
            y_real,
            aplicar_correspondencias(previsoes, correspondencias, peso),
        )
        for peso in PESOS_CORRESPONDENCIA
    }


def avaliar_lista_holdout(
    configuracoes,
    cache_matrizes,
    y_real,
    correspondencias,
    prefixo,
):
    resultados = []
    for numero, configuracao in enumerate(configuracoes, start=1):
        metricas = avaliar_no_corte(
            configuracao,
            cache_matrizes[configuracao["modo_bairro"]],
            y_real,
            correspondencias,
        )
        linha = deepcopy(configuracao)
        for peso, valor in metricas.items():
            linha[f"rmspe_bloco_corr_{int(peso * 100):02d}"] = valor
        resultados.append(linha)
        if numero % 20 == 0 or numero == len(configuracoes):
            print(f"{prefixo}: {numero}/{len(configuracoes)}")
    return pd.DataFrame(resultados)


def selecionar_melhores_por_modo(resultados, quantidade):
    return (
        resultados.sort_values("rmspe_bloco_corr_30")
        .groupby("modo_bairro", group_keys=False)
        .head(quantidade)
    )


raiz = os.path.join(os.path.dirname(__file__), "..")
df_original = pd.read_csv(
    os.path.join(raiz, "data", "conjunto_de_treinamento (5).csv")
)
df_base = df_original[
    ~df_original["Id"].isin([5910, 2405, 4568, 6004, 6654])
].copy()
df_corrigido = corrigir_dados(df_base, corrigir_alvo=True).reset_index(drop=True)

quantidade_bloco = round(len(df_corrigido) * 0.2)
ids_bloco = set(df_corrigido.nsmallest(quantidade_bloco, "Id")["Id"])
df_treino_bloco = df_corrigido[~df_corrigido["Id"].isin(ids_bloco)]
df_validacao_bloco = df_corrigido[df_corrigido["Id"].isin(ids_bloco)]
correspondencias_bloco = prever_correspondencias(
    df_treino_bloco,
    df_validacao_bloco,
)
cache_bloco = {
    modo: preparar_matrizes(df_treino_bloco, df_validacao_bloco, modo)
    for modo in MODOS_BAIRRO
}

configuracoes_estrutura = criar_busca_estrutura()
resultados_estrutura = avaliar_lista_holdout(
    configuracoes_estrutura,
    cache_bloco,
    df_validacao_bloco["preco"].to_numpy(),
    correspondencias_bloco,
    "Estruturas",
)
melhores_estruturas_df = selecionar_melhores_por_modo(
    resultados_estrutura,
    quantidade=2,
)
colunas_configuracao = list(configuracoes_estrutura[0].keys())
melhores_estruturas = melhores_estruturas_df[
    colunas_configuracao
].to_dict("records")

configuracoes_refinadas = criar_busca_refinada(melhores_estruturas)
chaves_existentes = {chave_configuracao(c) for c in configuracoes_estrutura}
configuracoes_refinadas = [
    c
    for c in configuracoes_refinadas
    if chave_configuracao(c) not in chaves_existentes
]
resultados_refinados = avaliar_lista_holdout(
    configuracoes_refinadas,
    cache_bloco,
    df_validacao_bloco["preco"].to_numpy(),
    correspondencias_bloco,
    "Refinamento",
)

resultados_holdout = pd.concat(
    [resultados_estrutura, resultados_refinados],
    ignore_index=True,
).drop_duplicates(subset=colunas_configuracao)
pre_selecionados = selecionar_melhores_por_modo(
    resultados_holdout,
    quantidade=5,
)
extras = resultados_holdout.sort_values("rmspe_bloco_corr_30").head(5)
pre_selecionados = pd.concat([pre_selecionados, extras]).drop_duplicates(
    subset=colunas_configuracao
)
configuracoes_cv = pre_selecionados[colunas_configuracao].to_dict("records")

folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(df_corrigido))
cache_folds = []
for numero_fold, (indices_treino, indices_validacao) in enumerate(folds, start=1):
    df_treino = df_corrigido.iloc[indices_treino]
    df_validacao = df_corrigido.iloc[indices_validacao]
    cache_folds.append(
        {
            "y_real": df_validacao["preco"].to_numpy(),
            "correspondencias": prever_correspondencias(
                df_treino,
                df_validacao,
            ),
            "matrizes": {
                modo: preparar_matrizes(df_treino, df_validacao, modo)
                for modo in MODOS_BAIRRO
            },
        }
    )
    print(f"Cache do fold {numero_fold}/5 preparado")

resultados_cv = []
for numero, configuracao in enumerate(configuracoes_cv, start=1):
    metricas_pesos = {peso: [] for peso in PESOS_CORRESPONDENCIA}
    for fold in cache_folds:
        metricas = avaliar_no_corte(
            configuracao,
            fold["matrizes"][configuracao["modo_bairro"]],
            fold["y_real"],
            fold["correspondencias"],
        )
        for peso, valor in metricas.items():
            metricas_pesos[peso].append(valor)

    linha = deepcopy(configuracao)
    for peso, valores in metricas_pesos.items():
        sufixo = int(peso * 100)
        linha[f"rmspe_kfold_corr_{sufixo:02d}"] = np.mean(valores)
        linha[f"desvio_kfold_corr_{sufixo:02d}"] = np.std(valores)
    resultados_cv.append(linha)
    print(f"Validacao completa: {numero}/{len(configuracoes_cv)}")

resultados_cv = pd.DataFrame(resultados_cv)
resultados_finais = pre_selecionados.merge(
    resultados_cv,
    on=colunas_configuracao,
    how="inner",
)
resultados_finais["criterio_selecao"] = (
    0.65 * resultados_finais["rmspe_bloco_corr_30"]
    + 0.35 * resultados_finais["rmspe_kfold_corr_30"]
)
resultados_finais = resultados_finais.sort_values("criterio_selecao")

pasta_resultados = os.path.join(raiz, "resultados")
os.makedirs(pasta_resultados, exist_ok=True)
resultados_estrutura.to_csv(
    os.path.join(pasta_resultados, "busca_xgboost_estruturas.csv"),
    index=False,
)
resultados_holdout.to_csv(
    os.path.join(pasta_resultados, "busca_xgboost_holdout.csv"),
    index=False,
)
resultados_finais.to_csv(
    os.path.join(pasta_resultados, "busca_xgboost_finalistas.csv"),
    index=False,
)

colunas_resumo = [
    "modo_bairro",
    "n_estimators",
    "learning_rate",
    "grow_policy",
    "max_depth",
    "max_leaves",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "gamma",
    "rmspe_bloco_corr_30",
    "rmspe_kfold_corr_30",
    "criterio_selecao",
]
print("\nMelhores finalistas:")
print(resultados_finais[colunas_resumo].head(15).to_string(index=False))
