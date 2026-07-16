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


def calcular_rmspe(y_real, y_previsto):
    erros_percentuais = (y_real - y_previsto) / y_real
    return np.sqrt(np.mean(erros_percentuais**2))


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


def corrigir_dados(df, corrigir_alvo=False):
    df = df.copy()
    df["area_util"] = df["area_util"].astype(float)
    df["area_extra"] = df["area_extra"].astype(float)

    mascara_area_util = (
        (df["tipo"] == "Apartamento")
        & (df["quartos"] > 0)
        & ((df["area_util"] / df["quartos"]) > 200)
    )
    df.loc[mascara_area_util, "area_util"] /= 10

    mascara_area_extra = (
        (df["tipo"] == "Apartamento")
        & (df["area_extra"] > 1000)
    )
    df.loc[mascara_area_extra, "area_extra"] /= 100

    if corrigir_alvo:
        df.loc[df["Id"].isin([2749, 4316]), "preco"] /= 10
        df.loc[df["Id"] == 6383, "vagas"] = 3

    return (
        df,
        set(df.loc[mascara_area_util, "Id"]),
        set(df.loc[mascara_area_extra, "Id"]),
    )


def preparar_modelo(df_treino, df_validacao):
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


def prever_modelo(df_treino, df_validacao):
    treino_modelo, validacao_modelo = preparar_modelo(
        df_treino,
        df_validacao,
    )
    modelo = criar_modelo()
    modelo.fit(
        treino_modelo.drop(columns=["Id", "preco"]),
        np.log1p(treino_modelo["preco"]),
        categorical_feature=["bairro"],
    )
    return np.expm1(
        modelo.predict(validacao_modelo.drop(columns=["Id", "preco"]))
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
        0.75 * previsoes_modelo[mascara]
        + 0.25 * previsoes_correspondencias[mascara]
    )
    return previsoes, mascara


raiz = os.path.join(os.path.dirname(__file__), "..")
df_original = pd.read_csv(
    os.path.join(raiz, "data", "conjunto_de_treinamento (5).csv")
)
df_teste_original = pd.read_csv(
    os.path.join(raiz, "data", "conjunto_de_teste (3).csv")
)

# O ID 6654 volta ao treino depois da correcao de area_extra.
df_base = df_original[
    ~df_original["Id"].isin([5910, 2405, 4568, 6004])
].copy()
df_corrigido, ids_area_util_treino, ids_area_extra_treino = corrigir_dados(
    df_base,
    corrigir_alvo=True,
)
df_teste_corrigido, ids_area_util_teste, ids_area_extra_teste = corrigir_dados(
    df_teste_original,
    corrigir_alvo=False,
)
df_corrigido = df_corrigido.reset_index(drop=True)

validacao_cruzada = KFold(n_splits=5, shuffle=True, random_state=42)
rmspe_modelo_folds = []
rmspe_combinado_folds = []

for indices_treino, indices_validacao in validacao_cruzada.split(df_corrigido):
    df_treino = df_corrigido.iloc[indices_treino]
    df_validacao = df_corrigido.iloc[indices_validacao]
    previsoes_modelo = prever_modelo(df_treino, df_validacao)
    previsoes_lookup = prever_correspondencias(df_treino, df_validacao)
    previsoes_combinadas, _ = combinar_previsoes(
        previsoes_modelo,
        previsoes_lookup,
    )
    y_validacao = df_validacao["preco"]
    rmspe_modelo_folds.append(calcular_rmspe(y_validacao, previsoes_modelo))
    rmspe_combinado_folds.append(
        calcular_rmspe(y_validacao, previsoes_combinadas)
    )

# Holdout com os 20% menores IDs.
quantidade_bloco = round(len(df_corrigido) * 0.2)
ids_bloco = set(df_corrigido.nsmallest(quantidade_bloco, "Id")["Id"])
df_treino_bloco = df_corrigido[~df_corrigido["Id"].isin(ids_bloco)]
df_validacao_bloco = df_corrigido[df_corrigido["Id"].isin(ids_bloco)]
previsoes_modelo_bloco = prever_modelo(df_treino_bloco, df_validacao_bloco)
previsoes_lookup_bloco = prever_correspondencias(
    df_treino_bloco,
    df_validacao_bloco,
)
previsoes_combinadas_bloco, _ = combinar_previsoes(
    previsoes_modelo_bloco,
    previsoes_lookup_bloco,
)
rmspe_modelo_bloco = calcular_rmspe(
    df_validacao_bloco["preco"],
    previsoes_modelo_bloco,
)
rmspe_combinado_bloco = calcular_rmspe(
    df_validacao_bloco["preco"],
    previsoes_combinadas_bloco,
)

# Treinamento final.
treino_modelo, teste_modelo = preparar_modelo(df_corrigido, df_teste_corrigido)
modelo_final = criar_modelo()
modelo_final.fit(
    treino_modelo.drop(columns=["Id", "preco"]),
    np.log1p(treino_modelo["preco"]),
    categorical_feature=["bairro"],
)
previsoes_modelo_teste = np.expm1(
    modelo_final.predict(teste_modelo.drop(columns=["Id"]))
)
previsoes_lookup_teste = prever_correspondencias(
    df_corrigido,
    df_teste_corrigido,
)
previsoes_finais, mascara_lookup_teste = combinar_previsoes(
    previsoes_modelo_teste,
    previsoes_lookup_teste,
)

submission = pd.read_csv(
    os.path.join(raiz, "data", "exemplo_arquivo_respostas.csv")
)
assert submission["Id"].equals(df_teste_original["Id"])
submission["preco"] = previsoes_finais.round(2)

pasta_saida = os.path.join(raiz, "submissions")
os.makedirs(pasta_saida, exist_ok=True)
caminho_saida = os.path.join(
    pasta_saida,
    "submission_lightgbm_area_extra_corrigida_correspondencia_25.csv",
)
submission.to_csv(caminho_saida, index=False)

assert ids_area_extra_treino == {3656, 6654}
assert ids_area_extra_teste == {1012, 1969, 1998}
assert submission.shape == (2000, 2)
assert submission["Id"].is_unique
assert not submission.isna().any().any()
assert not (submission["preco"] < 0).any()

print(f"Areas extras corrigidas no treino: {sorted(ids_area_extra_treino)}")
print(f"Areas extras corrigidas no teste: {sorted(ids_area_extra_teste)}")
print(f"KFold modelo: {np.mean(rmspe_modelo_folds) * 100:.2f}%")
print(f"KFold com correspondencias: {np.mean(rmspe_combinado_folds) * 100:.2f}%")
print(f"Holdout modelo: {rmspe_modelo_bloco * 100:.2f}%")
print(f"Holdout com correspondencias: {rmspe_combinado_bloco * 100:.2f}%")
print(f"Correspondencias no teste: {mascara_lookup_teste.sum()} de 2000")
print(f"Submission: {os.path.abspath(caminho_saida)}")
