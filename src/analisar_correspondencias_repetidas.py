import os

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

from pre_processamento import (
    criar_features_modelo_bairro_categorico,
    selecionar_bairros_frequentes,
)


def calcular_rmspe(y_real, y_previsto):
    erros_percentuais = (y_real - y_previsto) / y_real
    return np.sqrt(np.mean(erros_percentuais**2))


def corrigir_dados(df):
    df = df.copy()
    df["area_util"] = df["area_util"].astype(float)
    mascara_area = (
        (df["tipo"] == "Apartamento")
        & (df["quartos"] > 0)
        & ((df["area_util"] / df["quartos"]) > 200)
    )
    df.loc[mascara_area, "area_util"] /= 10

    if "preco" in df.columns:
        df.loc[df["Id"].isin([2749, 4316]), "preco"] /= 10
        df.loc[df["Id"] == 6383, "vagas"] = 3

    return df


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


def prever_modelo(df_treino, df_validacao):
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
    x_validacao = validacao_modelo.drop(columns=["Id", "preco"])

    modelo = criar_modelo()
    modelo.fit(
        x_treino,
        np.log1p(treino_modelo["preco"]),
        categorical_feature=["bairro"],
    )
    return np.expm1(modelo.predict(x_validacao))


def prever_por_correspondencia(df_treino, df_validacao, colunas_chave):
    tabela_precos = (
        df_treino.groupby(colunas_chave, dropna=False)["preco"]
        .median()
        .rename("preco_correspondencia")
        .reset_index()
    )
    validacao_com_ordem = df_validacao.reset_index(drop=True).reset_index(
        names="ordem_original"
    )
    correspondencias = validacao_com_ordem.merge(
        tabela_precos,
        on=colunas_chave,
        how="left",
    ).sort_values("ordem_original")
    return correspondencias["preco_correspondencia"].to_numpy()


raiz = os.path.join(os.path.dirname(__file__), "..")
df_original = pd.read_csv(
    os.path.join(raiz, "data", "conjunto_de_treinamento (5).csv")
)
df_original = df_original[
    ~df_original["Id"].isin([5910, 2405, 4568, 6004, 6654])
].copy()
df_corrigido = corrigir_dados(df_original).reset_index(drop=True)

colunas_completas = [
    coluna for coluna in df_corrigido.columns if coluna not in ["Id", "preco"]
]
colunas_sem_diferenciais = [
    coluna for coluna in colunas_completas if coluna != "diferenciais"
]
colunas_estruturais = [
    "tipo",
    "bairro",
    "area_util",
    "area_extra",
    "quartos",
    "suites",
    "vagas",
]
chaves = {
    "completa": colunas_completas,
    "sem_diferenciais": colunas_sem_diferenciais,
    "estrutural": colunas_estruturais,
}

validacao_cruzada = KFold(n_splits=5, shuffle=True, random_state=42)
divisoes = list(validacao_cruzada.split(df_corrigido))
previsoes_modelo = np.zeros(len(df_corrigido))
previsoes_lookup = {
    nome: np.full(len(df_corrigido), np.nan) for nome in chaves
}
fold_de_cada_linha = np.zeros(len(df_corrigido), dtype=int)

for numero_fold, (indices_treino, indices_validacao) in enumerate(
    divisoes,
    start=1,
):
    df_treino = df_corrigido.iloc[indices_treino]
    df_validacao = df_corrigido.iloc[indices_validacao]
    previsoes_modelo[indices_validacao] = prever_modelo(df_treino, df_validacao)
    fold_de_cada_linha[indices_validacao] = numero_fold

    for nome, colunas in chaves.items():
        previsoes_lookup[nome][indices_validacao] = prever_por_correspondencia(
            df_treino,
            df_validacao,
            colunas,
        )

resultados = []
y_real = df_corrigido["preco"].to_numpy()

for nome, previsoes_correspondencia in previsoes_lookup.items():
    mascara_correspondencia = ~np.isnan(previsoes_correspondencia)

    for peso_correspondencia in [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]:
        previsoes_finais = previsoes_modelo.copy()
        previsoes_finais[mascara_correspondencia] = (
            (1 - peso_correspondencia)
            * previsoes_modelo[mascara_correspondencia]
            + peso_correspondencia
            * previsoes_correspondencia[mascara_correspondencia]
        )

        rmspe_folds = []
        for numero_fold in range(1, 6):
            mascara_fold = fold_de_cada_linha == numero_fold
            rmspe_folds.append(
                calcular_rmspe(
                    y_real[mascara_fold],
                    previsoes_finais[mascara_fold],
                )
            )

        resultados.append(
            {
                "tipo_validacao": "kfold",
                "chave": nome,
                "peso_correspondencia": peso_correspondencia,
                "cobertura": mascara_correspondencia.mean(),
                "rmspe": np.mean(rmspe_folds),
                "desvio": np.std(rmspe_folds),
            }
        )

# Holdout com os 20% menores IDs.
quantidade_bloco = round(len(df_corrigido) * 0.2)
ids_bloco = set(df_corrigido.nsmallest(quantidade_bloco, "Id")["Id"])
df_treino_bloco = df_corrigido[~df_corrigido["Id"].isin(ids_bloco)]
df_validacao_bloco = df_corrigido[df_corrigido["Id"].isin(ids_bloco)]
previsoes_modelo_bloco = prever_modelo(df_treino_bloco, df_validacao_bloco)
y_bloco = df_validacao_bloco["preco"].to_numpy()

for nome, colunas in chaves.items():
    previsoes_correspondencia = prever_por_correspondencia(
        df_treino_bloco,
        df_validacao_bloco,
        colunas,
    )
    mascara_correspondencia = ~np.isnan(previsoes_correspondencia)

    for peso_correspondencia in [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]:
        previsoes_finais = previsoes_modelo_bloco.copy()
        previsoes_finais[mascara_correspondencia] = (
            (1 - peso_correspondencia)
            * previsoes_modelo_bloco[mascara_correspondencia]
            + peso_correspondencia
            * previsoes_correspondencia[mascara_correspondencia]
        )
        resultados.append(
            {
                "tipo_validacao": "bloco_id",
                "chave": nome,
                "peso_correspondencia": peso_correspondencia,
                "cobertura": mascara_correspondencia.mean(),
                "rmspe": calcular_rmspe(y_bloco, previsoes_finais),
                "desvio": np.nan,
            }
        )

resultados = pd.DataFrame(resultados)
pasta_resultados = os.path.join(raiz, "resultados")
os.makedirs(pasta_resultados, exist_ok=True)
resultados.to_csv(
    os.path.join(pasta_resultados, "comparacao_correspondencias_repetidas.csv"),
    index=False,
)

print("Melhores resultados por validacao:")
print(
    resultados.sort_values("rmspe")
    .groupby("tipo_validacao", as_index=False)
    .first()[
        [
            "tipo_validacao",
            "chave",
            "peso_correspondencia",
            "cobertura",
            "rmspe",
            "desvio",
        ]
    ]
    .to_string(index=False)
)
print("\nReferencias sem correspondencia:")
print(
    f"KFold: {np.mean([calcular_rmspe(y_real[fold_de_cada_linha == fold], previsoes_modelo[fold_de_cada_linha == fold]) for fold in range(1, 6)]) * 100:.2f}%"
)
print(f"Bloco por ID: {calcular_rmspe(y_bloco, previsoes_modelo_bloco) * 100:.2f}%")
