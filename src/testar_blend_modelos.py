import os

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

from pre_processamento import (
    criar_features_modelo_bairro_categorico,
    df,
    selecionar_bairros_frequentes,
    x,
    y,
)


def calcular_rmspe(y_real, y_previsto):
    erros_percentuais = (y_real - y_previsto) / y_real
    return np.sqrt(np.mean(erros_percentuais**2))


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


validacao_cruzada = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
divisoes = list(validacao_cruzada.split(df))

previsoes_one_hot = np.zeros(len(df))
previsoes_categorico = np.zeros(len(df))

for numero_fold, (indices_treino, indices_validacao) in enumerate(
    divisoes,
    start=1,
):
    modelo_one_hot = criar_modelo_one_hot()
    modelo_one_hot.fit(
        x.iloc[indices_treino],
        np.log1p(y.iloc[indices_treino]),
    )
    previsoes_one_hot[indices_validacao] = np.expm1(
        modelo_one_hot.predict(x.iloc[indices_validacao])
    )

    df_treino = df.iloc[indices_treino]
    df_validacao = df.iloc[indices_validacao]
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

    x_treino_cat = treino_modelo.drop(columns=["Id", "preco"])
    y_treino_cat = treino_modelo["preco"]
    x_validacao_cat = validacao_modelo.drop(columns=["Id", "preco"])

    modelo_categorico = criar_modelo_categorico()
    modelo_categorico.fit(
        x_treino_cat,
        np.log1p(y_treino_cat),
        categorical_feature=["bairro"],
    )
    previsoes_categorico[indices_validacao] = np.expm1(
        modelo_categorico.predict(x_validacao_cat)
    )

    print(f"Fold {numero_fold}/5 concluido")


def avaliar_previsoes_por_fold(previsoes):
    resultados = []
    for _, indices_validacao in divisoes:
        resultados.append(
            calcular_rmspe(y.iloc[indices_validacao], previsoes[indices_validacao])
        )
    return np.array(resultados)


resultados = []
pesos_categorico = np.arange(0.0, 1.01, 0.1)

for tipo_blend in ["aritmetico", "logaritmico"]:
    for peso_categorico in pesos_categorico:
        if tipo_blend == "aritmetico":
            previsoes_blend = (
                (1 - peso_categorico) * previsoes_one_hot
                + peso_categorico * previsoes_categorico
            )
        else:
            previsoes_blend = np.expm1(
                (1 - peso_categorico) * np.log1p(previsoes_one_hot)
                + peso_categorico * np.log1p(previsoes_categorico)
            )

        rmspe_folds = avaliar_previsoes_por_fold(previsoes_blend)
        resultados.append(
            {
                "tipo_blend": tipo_blend,
                "peso_categorico": peso_categorico,
                "peso_one_hot": 1 - peso_categorico,
                "rmspe_cv": rmspe_folds.mean(),
                "desvio_rmspe_cv": rmspe_folds.std(),
                **{
                    f"rmspe_fold_{indice}": valor
                    for indice, valor in enumerate(rmspe_folds, start=1)
                },
            }
        )

resultados = pd.DataFrame(resultados).sort_values("rmspe_cv")

raiz = os.path.join(os.path.dirname(__file__), "..")
pasta_resultados = os.path.join(raiz, "resultados")
os.makedirs(pasta_resultados, exist_ok=True)
caminho_resultados = os.path.join(
    pasta_resultados,
    "comparacao_blend_lightgbm.csv",
)
resultados.to_csv(caminho_resultados, index=False)

print("\nMelhores combinacoes:")
print(
    resultados[
        [
            "tipo_blend",
            "peso_one_hot",
            "peso_categorico",
            "rmspe_cv",
            "desvio_rmspe_cv",
        ]
    ].head(10).to_string(index=False)
)
print(f"\nResultados: {os.path.abspath(caminho_resultados)}")
