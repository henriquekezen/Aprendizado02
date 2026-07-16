import os

import numpy as np
import pandas as pd
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

CONFIGURACOES = [
    {
        "ordem": 1,
        "nome": "equilibrado",
        "arquivo": "submission_xgb_01_equilibrado_corr30.csv",
        "modo_bairro": "nativo",
        "n_estimators": 800,
        "learning_rate": 0.03,
        "grow_policy": "depthwise",
        "max_depth": 4,
        "max_leaves": 0,
        "min_child_weight": 12,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.2,
        "reg_lambda": 3.0,
        "gamma": 0.0,
        "max_cat_threshold": 64,
        "rmspe_bloco": 0.217691,
        "rmspe_kfold": 0.230592,
    },
    {
        "ordem": 2,
        "nome": "gamma",
        "arquivo": "submission_xgb_02_gamma_corr30.csv",
        "modo_bairro": "nativo",
        "n_estimators": 800,
        "learning_rate": 0.03,
        "grow_policy": "depthwise",
        "max_depth": 4,
        "max_leaves": 0,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "gamma": 0.1,
        "max_cat_threshold": 64,
        "rmspe_bloco": 0.217641,
        "rmspe_kfold": 0.230907,
    },
    {
        "ordem": 3,
        "nome": "1200_arvores",
        "arquivo": "submission_xgb_03_1200_arvores_corr30.csv",
        "modo_bairro": "nativo",
        "n_estimators": 1200,
        "learning_rate": 0.02,
        "grow_policy": "depthwise",
        "max_depth": 4,
        "max_leaves": 0,
        "min_child_weight": 12,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "gamma": 0.0,
        "max_cat_threshold": 64,
        "rmspe_bloco": 0.217751,
        "rmspe_kfold": 0.231216,
    },
    {
        "ordem": 4,
        "nome": "300_arvores_bloco_id",
        "arquivo": "submission_xgb_04_300_arvores_bloco_id_corr30.csv",
        "modo_bairro": "nativo",
        "n_estimators": 300,
        "learning_rate": 0.08,
        "grow_policy": "depthwise",
        "max_depth": 4,
        "max_leaves": 0,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "gamma": 0.0,
        "max_cat_threshold": 64,
        "rmspe_bloco": 0.216703,
        "rmspe_kfold": 0.235287,
    },
    {
        "ordem": 5,
        "nome": "bairro_target_m2",
        "arquivo": "submission_xgb_05_bairro_target_m2_corr30.csv",
        "modo_bairro": "target_m2",
        "n_estimators": 800,
        "learning_rate": 0.03,
        "grow_policy": "depthwise",
        "max_depth": 4,
        "max_leaves": 0,
        "min_child_weight": 12,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.2,
        "reg_lambda": 3.0,
        "gamma": 0.0,
        "max_cat_threshold": 64,
        "rmspe_bloco": 0.220614,
        "rmspe_kfold": 0.232444,
    },
]


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


def preparar_features(df_treino, df_teste, modo_bairro):
    df_treino = df_treino.reset_index(drop=True)
    df_teste = df_teste.reset_index(drop=True)
    bairros_mantidos = selecionar_bairros_frequentes(
        df_treino,
        minimo_imoveis=10,
    )
    treino = criar_features_modelo_bairro_categorico(
        df_treino,
        bairros_mantidos,
    )
    teste = criar_features_modelo_bairro_categorico(
        df_teste,
        bairros_mantidos,
    )

    if modo_bairro == "nativo":
        return treino, teste

    if modo_bairro != "target_m2":
        raise ValueError(f"Modo de bairro desconhecido: {modo_bairro}")

    bairro_treino = treino["bairro"].astype("string")
    bairro_teste = teste["bairro"].astype("string")
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
    teste["bairro_preco_m2_log"] = bairro_teste.map(
        estatisticas["media_suavizada"]
    ).fillna(media_global).astype(float)
    treino["bairro_frequencia"] = bairro_treino.map(
        estatisticas["frequencia"]
    ).astype(float)
    teste["bairro_frequencia"] = bairro_teste.map(
        estatisticas["frequencia"]
    ).fillna(0).astype(float)
    treino = treino.drop(columns=["bairro"])
    teste = teste.drop(columns=["bairro"])
    return treino, teste


def prever_correspondencias(df_treino, df_teste):
    tabela = (
        df_treino.groupby(COLUNAS_CORRESPONDENCIA, dropna=False)["preco"]
        .median()
        .rename("preco_correspondencia")
        .reset_index()
    )
    teste_ordenado = df_teste.reset_index(drop=True).reset_index(
        names="ordem_original"
    )
    return (
        teste_ordenado.merge(
            tabela,
            on=COLUNAS_CORRESPONDENCIA,
            how="left",
        )
        .sort_values("ordem_original")["preco_correspondencia"]
        .to_numpy()
    )


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


def aplicar_correspondencias(previsoes, correspondencias, peso=0.30):
    resultado = previsoes.copy()
    mascara = ~np.isnan(correspondencias)
    resultado[mascara] = (
        (1 - peso) * previsoes[mascara]
        + peso * correspondencias[mascara]
    )
    return resultado


raiz = os.path.join(os.path.dirname(__file__), "..")
df_original = pd.read_csv(
    os.path.join(raiz, "data", "conjunto_de_treinamento (5).csv")
)
df_teste_original = pd.read_csv(
    os.path.join(raiz, "data", "conjunto_de_teste (3).csv")
)
df_base = df_original[
    ~df_original["Id"].isin([5910, 2405, 4568, 6004, 6654])
].copy()
df_corrigido = corrigir_dados(df_base, corrigir_alvo=True).reset_index(drop=True)
df_teste_corrigido = corrigir_dados(
    df_teste_original,
    corrigir_alvo=False,
).reset_index(drop=True)

modos_necessarios = {config["modo_bairro"] for config in CONFIGURACOES}
cache_features = {}
for modo in modos_necessarios:
    treino, teste = preparar_features(df_corrigido, df_teste_corrigido, modo)
    cache_features[modo] = {
        "x_treino": treino.drop(columns=["Id", "preco"]),
        "y_treino_log": np.log1p(treino["preco"]),
        "x_teste": teste.drop(columns=["Id", "preco"], errors="ignore"),
    }

correspondencias_teste = prever_correspondencias(
    df_corrigido,
    df_teste_corrigido,
)
modelo_resposta = pd.read_csv(
    os.path.join(raiz, "data", "exemplo_arquivo_respostas.csv")
)
pasta_submissions = os.path.join(raiz, "submissions")
os.makedirs(pasta_submissions, exist_ok=True)

for configuracao in CONFIGURACOES:
    features = cache_features[configuracao["modo_bairro"]]
    modelo = criar_modelo(configuracao)
    modelo.fit(features["x_treino"], features["y_treino_log"])
    previsoes = np.expm1(modelo.predict(features["x_teste"]))
    previsoes = aplicar_correspondencias(
        previsoes,
        correspondencias_teste,
        peso=0.30,
    )

    submission = modelo_resposta.copy()
    submission["preco"] = previsoes.round(2)
    caminho = os.path.join(pasta_submissions, configuracao["arquivo"])
    submission.to_csv(caminho, index=False)

    assert submission.shape == (2000, 2)
    assert submission["Id"].is_unique
    assert not submission.isna().any().any()
    assert not (submission["preco"] <= 0).any()
    print(f"{configuracao['ordem']}: {os.path.abspath(caminho)}")

colunas_manifesto = [
    "ordem",
    "nome",
    "arquivo",
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
    "max_cat_threshold",
    "rmspe_bloco",
    "rmspe_kfold",
]
manifesto = pd.DataFrame(CONFIGURACOES)[colunas_manifesto]
pasta_resultados = os.path.join(raiz, "resultados")
os.makedirs(pasta_resultados, exist_ok=True)
manifesto.to_csv(
    os.path.join(pasta_resultados, "xgboost_submissions_escolhidas.csv"),
    index=False,
)
