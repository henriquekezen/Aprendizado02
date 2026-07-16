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
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def calcular_rmspe(y_real, y_previsto):
    erros_percentuais = (y_real - y_previsto) / y_real
    return np.sqrt(np.mean(erros_percentuais**2))


def avaliar_one_hot(divisoes):
    resultados_folds = []

    for indices_treino, indices_validacao in divisoes:
        modelo = criar_modelo()
        modelo.fit(
            x.iloc[indices_treino],
            np.log1p(y.iloc[indices_treino]),
        )
        previsoes = np.expm1(modelo.predict(x.iloc[indices_validacao]))
        resultados_folds.append(
            calcular_rmspe(y.iloc[indices_validacao], previsoes)
        )

    return resultados_folds, x.shape[1]


def avaliar_bairro_categorico(divisoes, minimo_imoveis):
    resultados_folds = []
    quantidades_features = []
    quantidades_bairros = []

    for indices_treino, indices_validacao in divisoes:
        df_treino = df.iloc[indices_treino]
        df_validacao = df.iloc[indices_validacao]

        bairros_mantidos = selecionar_bairros_frequentes(
            df_treino,
            minimo_imoveis=minimo_imoveis,
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

        assert x_treino.columns.equals(x_validacao.columns)
        assert x_treino["bairro"].dtype == x_validacao["bairro"].dtype

        modelo = criar_modelo()
        modelo.fit(
            x_treino,
            np.log1p(y_treino),
            categorical_feature=["bairro"],
        )

        previsoes = np.expm1(modelo.predict(x_validacao))
        resultados_folds.append(calcular_rmspe(y_validacao, previsoes))
        quantidades_features.append(x_treino.shape[1])
        quantidades_bairros.append(len(bairros_mantidos))

    return (
        resultados_folds,
        int(np.mean(quantidades_features)),
        float(np.mean(quantidades_bairros)),
    )


validacao_cruzada = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
divisoes = list(validacao_cruzada.split(df))

experimentos = []

rmspe_one_hot, n_features_one_hot = avaliar_one_hot(divisoes)
experimentos.append(
    {
        "estrategia": "one_hot_atual",
        "minimo_imoveis_bairro": np.nan,
        "n_features": n_features_one_hot,
        "bairros_mantidos_medio": 66,
        "rmspe_cv": np.mean(rmspe_one_hot),
        "desvio_rmspe_cv": np.std(rmspe_one_hot),
        **{
            f"rmspe_fold_{indice}": valor
            for indice, valor in enumerate(rmspe_one_hot, start=1)
        },
    }
)
print(f"one_hot_atual: {np.mean(rmspe_one_hot) * 100:.2f}%")

variantes_categoricas = [
    ("categorico_sem_agrupamento", None),
    ("categorico_min_5", 5),
    ("categorico_min_10", 10),
    ("categorico_min_15", 15),
    ("categorico_min_20", 20),
]

for nome, minimo_imoveis in variantes_categoricas:
    rmspe_folds, n_features, bairros_mantidos = avaliar_bairro_categorico(
        divisoes,
        minimo_imoveis,
    )
    experimentos.append(
        {
            "estrategia": nome,
            "minimo_imoveis_bairro": minimo_imoveis,
            "n_features": n_features,
            "bairros_mantidos_medio": bairros_mantidos,
            "rmspe_cv": np.mean(rmspe_folds),
            "desvio_rmspe_cv": np.std(rmspe_folds),
            **{
                f"rmspe_fold_{indice}": valor
                for indice, valor in enumerate(rmspe_folds, start=1)
            },
        }
    )
    print(
        f"{nome}: {np.mean(rmspe_folds) * 100:.2f}% "
        f"(+/- {np.std(rmspe_folds) * 100:.2f} p.p.)"
    )

resultados = pd.DataFrame(experimentos).sort_values("rmspe_cv")

raiz = os.path.join(os.path.dirname(__file__), "..")
pasta_resultados = os.path.join(raiz, "resultados")
os.makedirs(pasta_resultados, exist_ok=True)
caminho_resultados = os.path.join(
    pasta_resultados,
    "comparacao_bairro_categorico.csv",
)
resultados.to_csv(caminho_resultados, index=False)

print("\nResultado ordenado:")
print(
    resultados[
        [
            "estrategia",
            "rmspe_cv",
            "desvio_rmspe_cv",
            "n_features",
            "bairros_mantidos_medio",
        ]
    ].to_string(index=False)
)
print(f"\nResultados completos: {os.path.abspath(caminho_resultados)}")
