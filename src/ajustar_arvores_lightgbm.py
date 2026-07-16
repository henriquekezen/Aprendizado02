import os

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_validate

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

configuracoes = [
    (0.05, 300),
    (0.05, 500),
    (0.05, 800),
    (0.05, 1200),
    (0.05, 1600),
    (0.03, 500),
    (0.03, 800),
    (0.03, 1200),
    (0.03, 1600),
    (0.03, 2200),
    (0.02, 800),
    (0.02, 1200),
    (0.02, 1600),
    (0.02, 2200),
    (0.02, 3000),
    (0.01, 1200),
    (0.01, 1800),
    (0.01, 2400),
    (0.01, 3200),
    (0.01, 4000),
]

y_log = np.log1p(y)
resultados = []

for indice, (learning_rate, n_estimators) in enumerate(configuracoes, start=1):
    modelo = LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=15,
        max_depth=5,
        min_child_samples=20,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
        verbosity=-1,
    )

    avaliacao = cross_validate(
        modelo,
        x,
        y_log,
        scoring=rmspe_scorer,
        cv=validacao_cruzada,
        n_jobs=-1,
        return_train_score=True,
    )

    rmspe_cv = -avaliacao["test_score"]
    rmspe_treino = -avaliacao["train_score"]

    resultados.append(
        {
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "rmspe_cv": rmspe_cv.mean(),
            "desvio_rmspe_cv": rmspe_cv.std(),
            "rmspe_treino": rmspe_treino.mean(),
        }
    )

    print(
        f"[{indice:02d}/{len(configuracoes)}] "
        f"lr={learning_rate:.3f}, arvores={n_estimators}: "
        f"RMSPE={rmspe_cv.mean() * 100:.2f}%"
    )

resultados = pd.DataFrame(resultados).sort_values("rmspe_cv")

raiz = os.path.join(os.path.dirname(__file__), "..")
pasta_resultados = os.path.join(raiz, "resultados")
os.makedirs(pasta_resultados, exist_ok=True)

caminho_resultados = os.path.join(
    pasta_resultados,
    "busca_arvores_lightgbm.csv",
)
resultados.to_csv(caminho_resultados, index=False)

melhor = resultados.iloc[0]
print("\nMelhor configuracao:")
print(f"  learning_rate: {melhor['learning_rate']}")
print(f"  n_estimators: {int(melhor['n_estimators'])}")
print(f"  RMSPE medio: {melhor['rmspe_cv'] * 100:.2f}%")
print(f"  Desvio: {melhor['desvio_rmspe_cv'] * 100:.2f} p.p.")
print(f"  RMSPE de treino: {melhor['rmspe_treino'] * 100:.2f}%")
print(f"Resultados completos: {os.path.abspath(caminho_resultados)}")
