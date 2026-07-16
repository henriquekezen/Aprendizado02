import os

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
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


def criar_xgboost():
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=800,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=4,
        random_state=42,
        n_jobs=-1,
    )


def criar_lightgbm():
    return LGBMRegressor(
        objective="regression",
        n_estimators=400,
        learning_rate=0.06,
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


def prever_modelos(df_treino, df_validacao):
    treino_modelo, validacao_modelo = preparar_features(
        df_treino,
        df_validacao,
    )
    x_treino = treino_modelo.drop(columns=["Id", "preco"])
    y_treino_log = np.log1p(treino_modelo["preco"])
    x_validacao = validacao_modelo.drop(
        columns=["Id", "preco"],
        errors="ignore",
    )

    modelo_xgb = criar_xgboost()
    modelo_xgb.fit(x_treino, y_treino_log)
    previsoes_xgb = np.expm1(modelo_xgb.predict(x_validacao))

    modelo_lgbm = criar_lightgbm()
    modelo_lgbm.fit(
        x_treino,
        y_treino_log,
        categorical_feature=["bairro"],
    )
    previsoes_lgbm = np.expm1(modelo_lgbm.predict(x_validacao))

    return previsoes_xgb, previsoes_lgbm


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


def aplicar_correspondencias(previsoes_modelo, previsoes_correspondencias):
    previsoes = previsoes_modelo.copy()
    mascara = ~np.isnan(previsoes_correspondencias)
    previsoes[mascara] = (
        0.70 * previsoes_modelo[mascara]
        + 0.30 * previsoes_correspondencias[mascara]
    )
    return previsoes


def calcular_tres_previsoes(
    previsoes_xgb,
    previsoes_lgbm,
    previsoes_correspondencias,
):
    previsoes_blend = 0.50 * previsoes_xgb + 0.50 * previsoes_lgbm
    return {
        "xgboost": aplicar_correspondencias(
            previsoes_xgb,
            previsoes_correspondencias,
        ),
        "lightgbm": aplicar_correspondencias(
            previsoes_lgbm,
            previsoes_correspondencias,
        ),
        "blend_50": aplicar_correspondencias(
            previsoes_blend,
            previsoes_correspondencias,
        ),
    }


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
resultados_kfold = {
    "xgboost": [],
    "lightgbm": [],
    "blend_50": [],
}

for numero_fold, (indices_treino, indices_validacao) in enumerate(
    validacao_cruzada.split(df_corrigido),
    start=1,
):
    df_treino = df_corrigido.iloc[indices_treino]
    df_validacao = df_corrigido.iloc[indices_validacao]
    previsoes_xgb, previsoes_lgbm = prever_modelos(df_treino, df_validacao)
    previsoes_correspondencias = prever_correspondencias(
        df_treino,
        df_validacao,
    )
    previsoes = calcular_tres_previsoes(
        previsoes_xgb,
        previsoes_lgbm,
        previsoes_correspondencias,
    )

    for nome, valores in previsoes.items():
        resultados_kfold[nome].append(
            calcular_rmspe(df_validacao["preco"], valores)
        )
    print(f"Fold {numero_fold}/5 concluido")

# Holdout com os 20% menores IDs.
quantidade_bloco = round(len(df_corrigido) * 0.2)
ids_bloco = set(df_corrigido.nsmallest(quantidade_bloco, "Id")["Id"])
df_treino_bloco = df_corrigido[~df_corrigido["Id"].isin(ids_bloco)]
df_validacao_bloco = df_corrigido[df_corrigido["Id"].isin(ids_bloco)]
previsoes_xgb_bloco, previsoes_lgbm_bloco = prever_modelos(
    df_treino_bloco,
    df_validacao_bloco,
)
previsoes_correspondencias_bloco = prever_correspondencias(
    df_treino_bloco,
    df_validacao_bloco,
)
previsoes_bloco = calcular_tres_previsoes(
    previsoes_xgb_bloco,
    previsoes_lgbm_bloco,
    previsoes_correspondencias_bloco,
)
resultados_bloco = {
    nome: calcular_rmspe(df_validacao_bloco["preco"], valores)
    for nome, valores in previsoes_bloco.items()
}

# Treinamento final.
previsoes_xgb_teste, previsoes_lgbm_teste = prever_modelos(
    df_corrigido,
    df_teste_corrigido,
)
previsoes_correspondencias_teste = prever_correspondencias(
    df_corrigido,
    df_teste_corrigido,
)
previsoes_teste = calcular_tres_previsoes(
    previsoes_xgb_teste,
    previsoes_lgbm_teste,
    previsoes_correspondencias_teste,
)

modelo_resposta = pd.read_csv(
    os.path.join(raiz, "data", "exemplo_arquivo_respostas.csv")
)
pasta_saida = os.path.join(raiz, "submissions")
os.makedirs(pasta_saida, exist_ok=True)
arquivos = {
    "xgboost": "submission_xgboost_inicial_correspondencia_30.csv",
    "blend_50": "submission_blend_xgb_lgbm400_50_correspondencia_30.csv",
}

for nome, arquivo in arquivos.items():
    submission = modelo_resposta.copy()
    submission["preco"] = previsoes_teste[nome].round(2)
    caminho_saida = os.path.join(pasta_saida, arquivo)
    submission.to_csv(caminho_saida, index=False)

    assert submission.shape == (2000, 2)
    assert submission["Id"].is_unique
    assert not submission.isna().any().any()
    assert not (submission["preco"] < 0).any()
    print(f"Submission: {os.path.abspath(caminho_saida)}")

resultados = pd.DataFrame(
    [
        {
            "modelo": nome,
            "rmspe_kfold": np.mean(valores),
            "desvio_kfold": np.std(valores),
            "rmspe_bloco_id": resultados_bloco[nome],
        }
        for nome, valores in resultados_kfold.items()
    ]
).sort_values("rmspe_bloco_id")

pasta_resultados = os.path.join(raiz, "resultados")
os.makedirs(pasta_resultados, exist_ok=True)
resultados.to_csv(
    os.path.join(pasta_resultados, "comparacao_xgboost_inicial.csv"),
    index=False,
)

print("\nResultados:")
print(resultados.to_string(index=False))
