import json
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


def preparar_folds():
    validacao_cruzada = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    folds = []

    for indices_treino, indices_validacao in validacao_cruzada.split(df):
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

        folds.append(
            {
                "x_treino": treino_modelo.drop(columns=["Id", "preco"]),
                "y_treino": treino_modelo["preco"],
                "x_validacao": validacao_modelo.drop(columns=["Id", "preco"]),
                "y_validacao": validacao_modelo["preco"],
            }
        )

    return folds


def avaliar_configuracao(folds, parametros):
    rmspe_validacao = []
    rmspe_treino = []

    for fold in folds:
        modelo = LGBMRegressor(**parametros)
        modelo.fit(
            fold["x_treino"],
            np.log1p(fold["y_treino"]),
            categorical_feature=["bairro"],
        )

        previsoes_validacao = np.expm1(modelo.predict(fold["x_validacao"]))
        previsoes_treino = np.expm1(modelo.predict(fold["x_treino"]))

        rmspe_validacao.append(
            calcular_rmspe(fold["y_validacao"], previsoes_validacao)
        )
        rmspe_treino.append(
            calcular_rmspe(fold["y_treino"], previsoes_treino)
        )

    return np.array(rmspe_validacao), np.array(rmspe_treino)


parametros_atuais = {
    "objective": "regression",
    "n_estimators": 800,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 5,
    "min_child_samples": 20,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "cat_smooth": 10.0,
    "cat_l2": 10.0,
    "min_data_per_group": 100,
    "max_cat_threshold": 32,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

etapas = [
    ("cat_smooth", [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]),
    ("cat_l2", [0.0, 1.0, 5.0, 10.0, 20.0, 50.0]),
    ("min_data_per_group", [10, 20, 50, 100, 200]),
    ("max_cat_threshold", [8, 16, 32, 64]),
    ("n_estimators", [400, 600, 800, 1000, 1200, 1600]),
]

folds = preparar_folds()
cache = {}
resultados = []


def avaliar_com_cache(parametros):
    chave = tuple(sorted(parametros.items()))
    if chave not in cache:
        cache[chave] = avaliar_configuracao(folds, parametros)
    return cache[chave]


rmspe_atual, _ = avaliar_com_cache(parametros_atuais)
print(
    f"Referencia: {rmspe_atual.mean() * 100:.2f}% "
    f"(+/- {rmspe_atual.std() * 100:.2f} p.p.)"
)

for nome_parametro, valores in etapas:
    rmspe_antes, _ = avaliar_com_cache(parametros_atuais)
    candidatos = []

    print(f"\nEtapa: {nome_parametro}")
    for valor in valores:
        parametros_teste = parametros_atuais.copy()
        parametros_teste[nome_parametro] = valor
        rmspe_cv, rmspe_treino = avaliar_com_cache(parametros_teste)
        folds_vencidos = int(np.sum(rmspe_cv < rmspe_antes))

        registro = {
            "etapa": nome_parametro,
            "valor": valor,
            "rmspe_cv": rmspe_cv.mean(),
            "desvio_rmspe_cv": rmspe_cv.std(),
            "rmspe_treino": rmspe_treino.mean(),
            "folds_vencidos": folds_vencidos,
            **{
                f"rmspe_fold_{indice}": resultado
                for indice, resultado in enumerate(rmspe_cv, start=1)
            },
        }
        resultados.append(registro)
        candidatos.append((rmspe_cv.mean(), valor, rmspe_cv))

        print(
            f"  {valor}: {rmspe_cv.mean() * 100:.2f}% "
            f"| treino {rmspe_treino.mean() * 100:.2f}% "
            f"| venceu {folds_vencidos}/5"
        )

    _, melhor_valor, melhor_rmspe = min(candidatos, key=lambda item: item[0])
    melhora = rmspe_antes.mean() - melhor_rmspe.mean()

    if melhora > 0 and np.sum(melhor_rmspe < rmspe_antes) >= 3:
        parametros_atuais[nome_parametro] = melhor_valor
        print(
            f"  Escolhido: {melhor_valor} "
            f"(ganho de {melhora * 100:.2f} p.p.)"
        )
    else:
        print("  Nenhuma alternativa foi aceita; valor atual mantido.")

resultados = pd.DataFrame(resultados)
rmspe_final, rmspe_treino_final = avaliar_com_cache(parametros_atuais)

raiz = os.path.join(os.path.dirname(__file__), "..")
pasta_resultados = os.path.join(raiz, "resultados")
os.makedirs(pasta_resultados, exist_ok=True)

caminho_resultados = os.path.join(
    pasta_resultados,
    "otimizacao_sequencial_lightgbm_categorico.csv",
)
resultados.to_csv(caminho_resultados, index=False)

caminho_parametros = os.path.join(
    pasta_resultados,
    "parametros_lightgbm_categorico.json",
)
with open(caminho_parametros, "w", encoding="utf-8") as arquivo:
    json.dump(parametros_atuais, arquivo, indent=2)

print("\nConfiguracao final:")
for nome, valor in parametros_atuais.items():
    print(f"  {nome}: {valor}")
print(f"RMSPE final: {rmspe_final.mean() * 100:.2f}%")
print(f"Desvio final: {rmspe_final.std() * 100:.2f} p.p.")
print(f"RMSPE de treino: {rmspe_treino_final.mean() * 100:.2f}%")
print(f"Resultados: {os.path.abspath(caminho_resultados)}")
