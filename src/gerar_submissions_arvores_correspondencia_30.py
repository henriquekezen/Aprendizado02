import os

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

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
    {"nome": "400_lr006", "n_estimators": 400, "learning_rate": 0.06},
    {"nome": "600_lr004", "n_estimators": 600, "learning_rate": 0.04},
    {"nome": "1200_lr002", "n_estimators": 1200, "learning_rate": 0.02},
    {"nome": "1600_lr0015", "n_estimators": 1600, "learning_rate": 0.015},
]


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


def criar_modelo(configuracao):
    return LGBMRegressor(
        objective="regression",
        n_estimators=configuracao["n_estimators"],
        learning_rate=configuracao["learning_rate"],
        num_leaves=15,
        max_depth=5,
        min_child_samples=20,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        cat_smooth=10.0,
        cat_l2=10.0,
        min_data_per_group=100,
        max_cat_threshold=32,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def preparar_features(df_treino, df_validacao):
    bairros_mantidos = selecionar_bairros_frequentes(
        df_treino,
        minimo_imoveis=10,
    )
    treino_modelo = criar_features_modelo_bairro_categorico(
        df_treino,
        bairros_mantidos,
    )
    validacao_modelo = criar_features_modelo_bairro_categorico(
        df_validacao,
        bairros_mantidos,
    )
    return treino_modelo, validacao_modelo


def prever_modelo(df_treino, df_validacao, configuracao):
    treino_modelo, validacao_modelo = preparar_features(
        df_treino,
        df_validacao,
    )
    modelo = criar_modelo(configuracao)
    modelo.fit(
        treino_modelo.drop(columns=["Id", "preco"]),
        np.log1p(treino_modelo["preco"]),
        categorical_feature=["bairro"],
    )
    return np.expm1(
        modelo.predict(
            validacao_modelo.drop(
                columns=["Id", "preco"],
                errors="ignore",
            )
        )
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


def combinar_previsoes(previsoes_modelo, previsoes_correspondencias):
    previsoes = previsoes_modelo.copy()
    mascara = ~np.isnan(previsoes_correspondencias)
    previsoes[mascara] = (
        0.70 * previsoes_modelo[mascara]
        + 0.30 * previsoes_correspondencias[mascara]
    )
    return previsoes


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

validacao_cruzada = KFold(n_splits=5, shuffle=True, random_state=42)
divisoes = list(validacao_cruzada.split(df_corrigido))

folds = []
for indices_treino, indices_validacao in divisoes:
    df_treino = df_corrigido.iloc[indices_treino]
    df_validacao = df_corrigido.iloc[indices_validacao]
    folds.append(
        {
            "treino": df_treino,
            "validacao": df_validacao,
            "previsoes_correspondencias": prever_correspondencias(
                df_treino,
                df_validacao,
            ),
        }
    )

quantidade_bloco = round(len(df_corrigido) * 0.2)
ids_bloco = set(df_corrigido.nsmallest(quantidade_bloco, "Id")["Id"])
df_treino_bloco = df_corrigido[~df_corrigido["Id"].isin(ids_bloco)]
df_validacao_bloco = df_corrigido[df_corrigido["Id"].isin(ids_bloco)]
previsoes_correspondencias_bloco = prever_correspondencias(
    df_treino_bloco,
    df_validacao_bloco,
)

previsoes_correspondencias_teste = prever_correspondencias(
    df_corrigido,
    df_teste_corrigido,
)

modelo_resposta = pd.read_csv(
    os.path.join(raiz, "data", "exemplo_arquivo_respostas.csv")
)
pasta_saida = os.path.join(raiz, "submissions")
os.makedirs(pasta_saida, exist_ok=True)
resultados = []

for configuracao in CONFIGURACOES:
    rmspe_folds = []

    for fold in folds:
        previsoes_modelo = prever_modelo(
            fold["treino"],
            fold["validacao"],
            configuracao,
        )
        previsoes = combinar_previsoes(
            previsoes_modelo,
            fold["previsoes_correspondencias"],
        )
        rmspe_folds.append(
            calcular_rmspe(fold["validacao"]["preco"], previsoes)
        )

    previsoes_modelo_bloco = prever_modelo(
        df_treino_bloco,
        df_validacao_bloco,
        configuracao,
    )
    previsoes_bloco = combinar_previsoes(
        previsoes_modelo_bloco,
        previsoes_correspondencias_bloco,
    )
    rmspe_bloco = calcular_rmspe(
        df_validacao_bloco["preco"],
        previsoes_bloco,
    )

    previsoes_modelo_teste = prever_modelo(
        df_corrigido,
        df_teste_corrigido,
        configuracao,
    )
    previsoes_teste = combinar_previsoes(
        previsoes_modelo_teste,
        previsoes_correspondencias_teste,
    )

    submission = modelo_resposta.copy()
    submission["preco"] = previsoes_teste.round(2)
    caminho_saida = os.path.join(
        pasta_saida,
        f"submission_arvores_{configuracao['nome']}_correspondencia_30.csv",
    )
    submission.to_csv(caminho_saida, index=False)

    assert submission.shape == (2000, 2)
    assert submission["Id"].is_unique
    assert not submission.isna().any().any()
    assert not (submission["preco"] < 0).any()

    resultados.append(
        {
            **configuracao,
            "rmspe_kfold": np.mean(rmspe_folds),
            "desvio_kfold": np.std(rmspe_folds),
            "rmspe_bloco_id": rmspe_bloco,
            "arquivo": caminho_saida,
        }
    )
    print(
        f"{configuracao['nome']}: KFold={np.mean(rmspe_folds) * 100:.2f}% "
        f"| bloco={rmspe_bloco * 100:.2f}%"
    )

resultados = pd.DataFrame(resultados).sort_values("rmspe_bloco_id")
pasta_resultados = os.path.join(raiz, "resultados")
os.makedirs(pasta_resultados, exist_ok=True)
resultados.to_csv(
    os.path.join(
        pasta_resultados,
        "comparacao_arvores_correspondencia_30.csv",
    ),
    index=False,
)

print("\nResultados ordenados pelo holdout por ID:")
print(
    resultados[
        [
            "nome",
            "n_estimators",
            "learning_rate",
            "rmspe_kfold",
            "desvio_kfold",
            "rmspe_bloco_id",
        ]
    ].to_string(index=False)
)
