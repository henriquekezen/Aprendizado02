import os

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from pre_processamento import (
    criar_features_modelo_bairro_categorico,
    df,
    selecionar_bairros_frequentes,
)


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

df_teste = pd.read_csv(caminho_teste)
ids_teste = df_teste["Id"].copy()

bairros_mantidos = selecionar_bairros_frequentes(
    df,
    minimo_imoveis=10,
)

df_modelo = criar_features_modelo_bairro_categorico(
    df,
    bairros_mantidos,
)
df_teste_modelo = criar_features_modelo_bairro_categorico(
    df_teste,
    bairros_mantidos,
)

y = df_modelo["preco"].copy()
x = df_modelo.drop(columns=["Id", "preco"])
x_teste = df_teste_modelo.drop(columns=["Id"])

assert x.columns.equals(x_teste.columns)
assert x["bairro"].dtype == x_teste["bairro"].dtype

modelo = LGBMRegressor(
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

modelo.fit(
    x,
    np.log1p(y),
    categorical_feature=["bairro"],
)

previsoes = np.expm1(modelo.predict(x_teste))

submission = pd.read_csv(caminho_exemplo)
assert submission["Id"].equals(ids_teste)
submission["preco"] = previsoes.round(2)

pasta_saida = os.path.join(raiz, "submissions")
os.makedirs(pasta_saida, exist_ok=True)
caminho_saida = os.path.join(
    pasta_saida,
    "submission_lightgbm_bairro_categorico.csv",
)
submission.to_csv(caminho_saida, index=False)

assert submission.shape == (len(df_teste), 2)
assert submission["Id"].is_unique
assert not submission.isna().any().any()
assert not (submission["preco"] < 0).any()

print(f"Bairros mantidos: {len(bairros_mantidos)}")
print(f"Features do modelo: {x.shape[1]}")
print(f"Imoveis de treino em Outros: {(df_modelo['bairro'] == 'Outros').sum()}")
print(f"Imoveis de teste em Outros: {(df_teste_modelo['bairro'] == 'Outros').sum()}")
print(f"Submission: {os.path.abspath(caminho_saida)}")
