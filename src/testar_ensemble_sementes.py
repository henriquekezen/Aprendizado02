import os

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

from pre_processamento import (
    criar_features_modelo_bairro_categorico,
    df,
    selecionar_bairros_frequentes,
)


def calcular_rmspe(y_real, y_previsto):
    erros_percentuais = (y_real - y_previsto) / y_real
    return np.sqrt(np.mean(erros_percentuais**2))


sementes = [42, 52, 62, 72, 82]
validacao_cruzada = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
divisoes = list(validacao_cruzada.split(df))

previsoes_log = np.zeros((len(sementes), len(df)))
y_completo = df["preco"].to_numpy()

for numero_fold, (indices_treino, indices_validacao) in enumerate(
    divisoes,
    start=1,
):
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

    x_treino = treino_modelo.drop(columns=["Id", "preco"])
    y_treino = treino_modelo["preco"]
    x_validacao = validacao_modelo.drop(columns=["Id", "preco"])

    for indice_semente, semente in enumerate(sementes):
        modelo = LGBMRegressor(
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
            random_state=semente,
            n_jobs=-1,
            verbosity=-1,
        )
        modelo.fit(
            x_treino,
            np.log1p(y_treino),
            categorical_feature=["bairro"],
        )
        previsoes_log[indice_semente, indices_validacao] = modelo.predict(
            x_validacao
        )

    print(f"Fold {numero_fold}/5 concluido")


def avaliar_por_fold(previsoes):
    resultados = []
    for _, indices_validacao in divisoes:
        resultados.append(
            calcular_rmspe(
                y_completo[indices_validacao],
                previsoes[indices_validacao],
            )
        )
    return np.array(resultados)


resultados = []

for quantidade in range(1, len(sementes) + 1):
    previsoes_ensemble = np.expm1(previsoes_log[:quantidade].mean(axis=0))
    rmspe_folds = avaliar_por_fold(previsoes_ensemble)
    resultados.append(
        {
            "quantidade_modelos": quantidade,
            "sementes": ",".join(map(str, sementes[:quantidade])),
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
    "comparacao_ensemble_sementes.csv",
)
resultados.to_csv(caminho_resultados, index=False)

print("\nResultados:")
print(
    resultados[
        ["quantidade_modelos", "sementes", "rmspe_cv", "desvio_rmspe_cv"]
    ].to_string(index=False)
)
print(f"\nResultados completos: {os.path.abspath(caminho_resultados)}")
