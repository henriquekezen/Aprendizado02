import os

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from pre_processamento import (
    criar_features_modelo,
    criar_features_modelo_bairro_categorico,
    df,
    selecionar_bairros_frequentes,
    x,
    y,
)


def criar_modelo_categorico():
    return LGBMRegressor(
        objective="regression",
        n_estimators=600,
        learning_rate=0.03,
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


def criar_modelo_one_hot():
    return LGBMRegressor(
        objective="regression",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=5,
        min_child_samples=20,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def salvar_submission(modelo_resposta, previsoes, caminho_saida):
    submission = modelo_resposta.copy()
    submission["preco"] = previsoes.round(2)

    assert submission.shape == modelo_resposta.shape
    assert submission["Id"].is_unique
    assert not submission.isna().any().any()
    assert not (submission["preco"] < 0).any()

    submission.to_csv(caminho_saida, index=False)
    print(f"Submission: {os.path.abspath(caminho_saida)}")


raiz = os.path.join(os.path.dirname(__file__), "..")
caminho_teste = os.path.join(
    raiz,
    "data",
    "conjunto_de_teste (3).csv",
)
caminho_exemplo = os.path.join(
    raiz,
    "data",
    "exemplo_arquivo_respostas.csv",
)
pasta_saida = os.path.join(raiz, "submissions")
os.makedirs(pasta_saida, exist_ok=True)

df_teste = pd.read_csv(caminho_teste)
modelo_resposta = pd.read_csv(caminho_exemplo)
assert modelo_resposta["Id"].equals(df_teste["Id"])

# Modelo com bairro categórico e 600 arvores.
bairros_mantidos = selecionar_bairros_frequentes(
    df,
    minimo_imoveis=10,
)
df_categorico = criar_features_modelo_bairro_categorico(
    df,
    bairros_mantidos,
)
df_teste_categorico = criar_features_modelo_bairro_categorico(
    df_teste,
    bairros_mantidos,
)

x_categorico = df_categorico.drop(columns=["Id", "preco"])
y_categorico = df_categorico["preco"]
x_teste_categorico = df_teste_categorico.drop(columns=["Id"])

modelo_categorico = criar_modelo_categorico()
modelo_categorico.fit(
    x_categorico,
    np.log1p(y_categorico),
    categorical_feature=["bairro"],
)
previsoes_categorico_log = modelo_categorico.predict(x_teste_categorico)
previsoes_categorico = np.expm1(previsoes_categorico_log)

caminho_categorico = os.path.join(
    pasta_saida,
    "submission_lightgbm_bairro_categorico_600.csv",
)
salvar_submission(
    modelo_resposta,
    previsoes_categorico,
    caminho_categorico,
)

# Blend: 90% do categorico e 10% do one-hot, combinados na escala de log.
df_teste_one_hot = criar_features_modelo(df_teste)
x_teste_one_hot = df_teste_one_hot.drop(columns=["Id"])
x_teste_one_hot = x_teste_one_hot.reindex(columns=x.columns, fill_value=0)

modelo_one_hot = criar_modelo_one_hot()
modelo_one_hot.fit(x, np.log1p(y))
previsoes_one_hot_log = modelo_one_hot.predict(x_teste_one_hot)

previsoes_blend = np.expm1(
    0.9 * previsoes_categorico_log
    + 0.1 * previsoes_one_hot_log
)

caminho_blend = os.path.join(
    pasta_saida,
    "submission_blend_90cat_10onehot.csv",
)
salvar_submission(
    modelo_resposta,
    previsoes_blend,
    caminho_blend,
)
