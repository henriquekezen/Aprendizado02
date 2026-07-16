import os

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

from pre_processamento import (
    criar_features_modelo_bairro_categorico,
    selecionar_bairros_frequentes,
)


def criar_modelo():
    return LGBMRegressor(
        objective="regression",
        n_estimators=800,
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


def calcular_rmspe(y_real, y_previsto):
    erros_percentuais = (y_real - y_previsto) / y_real
    return np.sqrt(np.mean(erros_percentuais**2))


raiz = os.path.join(os.path.dirname(__file__), "..")
caminho_treino = os.path.join(
    raiz,
    "data",
    "conjunto_de_treinamento (5).csv",
)
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

df_original = pd.read_csv(caminho_treino)
df_teste = pd.read_csv(caminho_teste)

# Somente erros extremos e praticamente incontestaveis.
ids_erros_incontestaveis = [
    5910,  # preco de R$ 750
    2405,  # preco de R$ 630 milhoes
    4568,  # preco de R$ 65 milhoes para 36 m2
    6004,  # preco de R$ 340 milhoes para 72 m2
    6654,  # 17.450 m2 de area extra
    6383,  # 30 vagas
]

df_quase_completo = df_original[
    ~df_original["Id"].isin(ids_erros_incontestaveis)
].copy()

validacao_cruzada = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
rmspe_folds = []

for numero_fold, (indices_treino, indices_validacao) in enumerate(
    validacao_cruzada.split(df_quase_completo),
    start=1,
):
    df_treino_fold = df_quase_completo.iloc[indices_treino]
    df_validacao_fold = df_quase_completo.iloc[indices_validacao]
    bairros_mantidos_fold = selecionar_bairros_frequentes(
        df_treino_fold,
        minimo_imoveis=10,
    )

    treino_modelo = criar_features_modelo_bairro_categorico(
        df_treino_fold,
        bairros_mantidos_fold,
    )
    validacao_modelo = criar_features_modelo_bairro_categorico(
        df_validacao_fold,
        bairros_mantidos_fold,
    )

    x_treino = treino_modelo.drop(columns=["Id", "preco"])
    y_treino = treino_modelo["preco"]
    x_validacao = validacao_modelo.drop(columns=["Id", "preco"])
    y_validacao = validacao_modelo["preco"]

    modelo = criar_modelo()
    modelo.fit(
        x_treino,
        np.log1p(y_treino),
        categorical_feature=["bairro"],
    )
    previsoes = np.expm1(modelo.predict(x_validacao))
    rmspe_fold = calcular_rmspe(y_validacao, previsoes)
    rmspe_folds.append(rmspe_fold)
    print(f"Fold {numero_fold}: {rmspe_fold * 100:.2f}%")

bairros_mantidos = selecionar_bairros_frequentes(
    df_quase_completo,
    minimo_imoveis=10,
)
df_modelo = criar_features_modelo_bairro_categorico(
    df_quase_completo,
    bairros_mantidos,
)
df_teste_modelo = criar_features_modelo_bairro_categorico(
    df_teste,
    bairros_mantidos,
)

x = df_modelo.drop(columns=["Id", "preco"])
y = df_modelo["preco"]
x_teste = df_teste_modelo.drop(columns=["Id"])

modelo_final = criar_modelo()
modelo_final.fit(
    x,
    np.log1p(y),
    categorical_feature=["bairro"],
)
previsoes_finais = np.expm1(modelo_final.predict(x_teste))

submission = pd.read_csv(caminho_exemplo)
assert submission["Id"].equals(df_teste["Id"])
submission["preco"] = previsoes_finais.round(2)

pasta_saida = os.path.join(raiz, "submissions")
os.makedirs(pasta_saida, exist_ok=True)
caminho_saida = os.path.join(
    pasta_saida,
    "submission_lightgbm_categorico_800_quase_todos_dados.csv",
)
submission.to_csv(caminho_saida, index=False)

assert len(df_quase_completo) == len(df_original) - len(ids_erros_incontestaveis)
assert submission.shape == (len(df_teste), 2)
assert submission["Id"].is_unique
assert not submission.isna().any().any()
assert not (submission["preco"] < 0).any()

rmspe_folds = np.array(rmspe_folds)
print(f"\nLinhas utilizadas: {len(df_quase_completo)}")
print(f"Linhas removidas: {len(ids_erros_incontestaveis)}")
print(f"RMSPE medio: {rmspe_folds.mean() * 100:.2f}%")
print(f"Desvio: {rmspe_folds.std() * 100:.2f} p.p.")
print(f"Submission: {os.path.abspath(caminho_saida)}")
