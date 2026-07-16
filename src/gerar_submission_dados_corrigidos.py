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


def corrigir_erros_escala(df, corrigir_alvo=False):
    df_corrigido = df.copy()
    df_corrigido["area_util"] = df_corrigido["area_util"].astype(float)

    mascara_area = (
        (df_corrigido["tipo"] == "Apartamento")
        & (df_corrigido["quartos"] > 0)
        & ((df_corrigido["area_util"] / df_corrigido["quartos"]) > 200)
    )
    df_corrigido.loc[mascara_area, "area_util"] = (
        df_corrigido.loc[mascara_area, "area_util"] / 10
    )

    if corrigir_alvo:
        ids_preco_com_zero_extra = [2749, 4316]
        df_corrigido.loc[
            df_corrigido["Id"].isin(ids_preco_com_zero_extra),
            "preco",
        ] /= 10

        df_corrigido.loc[df_corrigido["Id"] == 6383, "vagas"] = 3

    return df_corrigido, set(df.loc[mascara_area, "Id"])


def treinar_prever(df_treino, df_validacao):
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
    return y_validacao, previsoes


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
df_teste_original = pd.read_csv(caminho_teste)

ids_sem_correcao_confiavel = [5910, 2405, 4568, 6004, 6654]
df_treino_base = df_original[
    ~df_original["Id"].isin(ids_sem_correcao_confiavel)
].copy()

df_corrigido, ids_area_treino = corrigir_erros_escala(
    df_treino_base,
    corrigir_alvo=True,
)
df_teste_corrigido, ids_area_teste = corrigir_erros_escala(
    df_teste_original,
    corrigir_alvo=False,
)

# Validacao cruzada aleatoria com todos os valores corrigidos.
validacao_cruzada = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
rmspe_folds = []

for indices_treino, indices_validacao in validacao_cruzada.split(df_corrigido):
    y_validacao, previsoes = treinar_prever(
        df_corrigido.iloc[indices_treino],
        df_corrigido.iloc[indices_validacao],
    )
    rmspe_folds.append(calcular_rmspe(y_validacao, previsoes))

# Holdout com os 20% menores IDs, aproximando a direcao do Kaggle.
quantidade_bloco = round(len(df_corrigido) * 0.2)
ids_bloco = set(df_corrigido.nsmallest(quantidade_bloco, "Id")["Id"])
treino_bloco = df_corrigido[~df_corrigido["Id"].isin(ids_bloco)]
validacao_bloco = df_corrigido[df_corrigido["Id"].isin(ids_bloco)]
y_bloco, previsoes_bloco = treinar_prever(treino_bloco, validacao_bloco)
rmspe_bloco = calcular_rmspe(y_bloco, previsoes_bloco)

# Treinamento final e submission.
bairros_mantidos = selecionar_bairros_frequentes(
    df_corrigido,
    minimo_imoveis=10,
)
df_modelo = criar_features_modelo_bairro_categorico(
    df_corrigido,
    bairros_mantidos,
)
df_teste_modelo = criar_features_modelo_bairro_categorico(
    df_teste_corrigido,
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
assert submission["Id"].equals(df_teste_original["Id"])
submission["preco"] = previsoes_finais.round(2)

pasta_saida = os.path.join(raiz, "submissions")
os.makedirs(pasta_saida, exist_ok=True)
caminho_saida = os.path.join(
    pasta_saida,
    "submission_lightgbm_categorico_800_dados_corrigidos.csv",
)
submission.to_csv(caminho_saida, index=False)

# Usa parcialmente a mediana de registros estruturalmente correspondentes.
colunas_correspondencia = [
    "tipo",
    "bairro",
    "area_util",
    "area_extra",
    "quartos",
    "suites",
    "vagas",
]
tabela_correspondencias = (
    df_corrigido.groupby(colunas_correspondencia, dropna=False)["preco"]
    .median()
    .rename("preco_correspondencia")
    .reset_index()
)
teste_com_ordem = df_teste_corrigido.reset_index(drop=True).reset_index(
    names="ordem_original"
)
teste_com_correspondencias = teste_com_ordem.merge(
    tabela_correspondencias,
    on=colunas_correspondencia,
    how="left",
).sort_values("ordem_original")
precos_correspondentes = teste_com_correspondencias[
    "preco_correspondencia"
].to_numpy()
mascara_correspondencia = ~np.isnan(precos_correspondentes)

previsoes_com_correspondencia = previsoes_finais.copy()
previsoes_com_correspondencia[mascara_correspondencia] = (
    0.75 * previsoes_finais[mascara_correspondencia]
    + 0.25 * precos_correspondentes[mascara_correspondencia]
)

submission_correspondencias = pd.read_csv(caminho_exemplo)
submission_correspondencias["preco"] = previsoes_com_correspondencia.round(2)
caminho_saida_correspondencias = os.path.join(
    pasta_saida,
    "submission_lightgbm_corrigido_correspondencia_25.csv",
)
submission_correspondencias.to_csv(
    caminho_saida_correspondencias,
    index=False,
)

submissions_pesos = {}
for peso_correspondencia in [0.10, 0.20, 0.30, 0.40]:
    previsoes_peso = previsoes_finais.copy()
    previsoes_peso[mascara_correspondencia] = (
        (1 - peso_correspondencia)
        * previsoes_finais[mascara_correspondencia]
        + peso_correspondencia
        * precos_correspondentes[mascara_correspondencia]
    )

    peso_percentual = round(peso_correspondencia * 100)
    submission_peso = pd.read_csv(caminho_exemplo)
    submission_peso["preco"] = previsoes_peso.round(2)
    caminho_submission_peso = os.path.join(
        pasta_saida,
        f"submission_correspondencia_{peso_percentual}.csv",
    )
    submission_peso.to_csv(caminho_submission_peso, index=False)
    submissions_pesos[peso_percentual] = (
        submission_peso,
        caminho_submission_peso,
    )

assert len(df_corrigido) == len(df_original) - len(ids_sem_correcao_confiavel)
assert ids_area_treino == {
    2101,
    2705,
    2865,
    3536,
    3553,
    3586,
    3604,
    3724,
    3937,
    3984,
    3998,
    4075,
    4220,
    4280,
    4708,
    4808,
    4820,
    5540,
    5575,
    6361,
    6606,
}
assert ids_area_teste == {228, 570, 1245}
assert submission.shape == (len(df_teste_original), 2)
assert submission["Id"].is_unique
assert not submission.isna().any().any()
assert not (submission["preco"] < 0).any()
assert submission_correspondencias.shape == submission.shape
assert submission_correspondencias["Id"].equals(submission["Id"])
assert not submission_correspondencias.isna().any().any()
assert not (submission_correspondencias["preco"] < 0).any()
for submission_peso, _ in submissions_pesos.values():
    assert submission_peso.shape == submission.shape
    assert submission_peso["Id"].equals(submission["Id"])
    assert not submission_peso.isna().any().any()
    assert not (submission_peso["preco"] < 0).any()

rmspe_folds = np.array(rmspe_folds)
print(f"Linhas utilizadas: {len(df_corrigido)}")
print(f"Areas corrigidas no treino: {len(ids_area_treino)}")
print(f"Areas corrigidas no teste: {sorted(ids_area_teste)}")
print("Precos corrigidos no treino: [2749, 4316]")
print("Vagas corrigidas no treino: ID 6383, de 30 para 3")
print(f"RMSPE CV: {rmspe_folds.mean() * 100:.2f}%")
print(f"Desvio CV: {rmspe_folds.std() * 100:.2f} p.p.")
print(f"RMSPE holdout por ID: {rmspe_bloco * 100:.2f}%")
print(f"Submission: {os.path.abspath(caminho_saida)}")
print(
    "Correspondencias no teste: "
    f"{mascara_correspondencia.sum()} de {len(df_teste_corrigido)}"
)
print(
    "Submission com correspondencias: "
    f"{os.path.abspath(caminho_saida_correspondencias)}"
)
for peso_percentual, (_, caminho_submission_peso) in submissions_pesos.items():
    print(
        f"Submission com peso {peso_percentual}%: "
        f"{os.path.abspath(caminho_submission_peso)}"
    )
