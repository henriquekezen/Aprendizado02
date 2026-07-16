import os

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate

from pre_processamento import x, y


def calcular_rmspe_em_log(y_real_log, y_previsto_log):
    y_real = np.expm1(y_real_log)
    y_previsto = np.expm1(y_previsto_log)
    erros_percentuais = (y_real - y_previsto) / y_real
    return np.sqrt(np.mean(erros_percentuais**2))


rmspe_scorer = make_scorer(
    calcular_rmspe_em_log,
    greater_is_better=False,
)

validacao_cruzada = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

y_log = np.log1p(y)

parametros_atuais = {
    "objective": "regression",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 5,
    "min_child_samples": 20,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": 1,
    "verbosity": -1,
}

modelo_atual = LGBMRegressor(**parametros_atuais)
resultado_atual = cross_validate(
    modelo_atual,
    x,
    y_log,
    scoring=rmspe_scorer,
    cv=validacao_cruzada,
    n_jobs=-1,
)

rmspe_atual = -resultado_atual["test_score"]
print(f"RMSPE medio atual: {rmspe_atual.mean() * 100:.2f}%")
print(f"Desvio atual: {rmspe_atual.std() * 100:.2f} p.p.")

modelo_busca = LGBMRegressor(
    objective="regression",
    random_state=42,
    n_jobs=1,
    verbosity=-1,
    subsample_freq=1,
)

espaco_busca = {
    "n_estimators": [300, 450, 600, 800, 1000],
    "learning_rate": [0.015, 0.025, 0.035, 0.05, 0.07],
    "num_leaves": [7, 12, 15, 20, 31, 45],
    "max_depth": [3, 4, 5, 6, 7, -1],
    "min_child_samples": [10, 20, 30, 45, 70, 100],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "subsample": [0.65, 0.75, 0.85, 0.95, 1.0],
    "reg_alpha": [0.0, 0.01, 0.05, 0.1, 0.3, 1.0],
    "reg_lambda": [0.0, 0.1, 0.5, 1.0, 3.0, 10.0],
}

busca = RandomizedSearchCV(
    estimator=modelo_busca,
    param_distributions=espaco_busca,
    n_iter=40,
    scoring=rmspe_scorer,
    cv=validacao_cruzada,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
    refit=True,
)

busca.fit(x, y_log)

resultados = pd.DataFrame(busca.cv_results_)
resultados["rmspe_cv"] = -resultados["mean_test_score"]
resultados["desvio_rmspe_cv"] = resultados["std_test_score"]
resultados["rmspe_treino"] = -resultados["mean_train_score"]
resultados = resultados.sort_values("rmspe_cv")

raiz = os.path.join(os.path.dirname(__file__), "..")
pasta_resultados = os.path.join(raiz, "resultados")
os.makedirs(pasta_resultados, exist_ok=True)

caminho_resultados = os.path.join(
    pasta_resultados,
    "busca_hiperparametros_lightgbm.csv",
)
resultados.to_csv(caminho_resultados, index=False)

print(f"Melhor RMSPE medio: {-busca.best_score_ * 100:.2f}%")
print("Melhores parametros:")
for nome, valor in busca.best_params_.items():
    print(f"  {nome}: {valor}")
print(f"Resultados completos: {os.path.abspath(caminho_resultados)}")
