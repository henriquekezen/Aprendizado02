"""Repete o KFold dos alphas finalistas com diferentes sementes."""

import os

import pandas as pd
from sklearn.model_selection import KFold

from testar_alphas_sample_weights import (
    MODELOS,
    avaliar_alpha,
    avaliar_holdout_id,
    carregar_dados,
)


ALPHAS = [1.25, 1.50, 1.75]
SEMENTES = [7, 42, 2026]


def main():
    raiz, df, _ = carregar_dados()
    resultados_folds = []

    for semente in SEMENTES:
        folds = list(
            KFold(
                n_splits=5,
                shuffle=True,
                random_state=semente,
            ).split(df)
        )
        for alpha in ALPHAS:
            print(f"\nSemente={semente}, alpha={alpha:.2f}")
            _, metricas = avaliar_alpha(df, alpha, folds)
            for modelo in MODELOS:
                for numero_fold, rmspe in enumerate(metricas[modelo], start=1):
                    resultados_folds.append(
                        {
                            "semente": semente,
                            "alpha": alpha,
                            "modelo": modelo,
                            "fold": numero_fold,
                            "rmspe": rmspe,
                        }
                    )

    detalhado = pd.DataFrame(resultados_folds)
    por_semente = (
        detalhado.groupby(["alpha", "modelo", "semente"], as_index=False)
        .agg(
            rmspe_medio=("rmspe", "mean"),
            desvio_folds=("rmspe", "std"),
        )
    )
    resumo = (
        detalhado.groupby(["alpha", "modelo"], as_index=False)
        .agg(
            rmspe_medio_15_folds=("rmspe", "mean"),
            desvio_15_folds=("rmspe", "std"),
        )
    )
    estabilidade = (
        por_semente.groupby(["alpha", "modelo"], as_index=False)
        .agg(
            melhor_media_semente=("rmspe_medio", "min"),
            pior_media_semente=("rmspe_medio", "max"),
            amplitude_entre_sementes=("rmspe_medio", lambda x: x.max() - x.min()),
        )
    )
    resumo = resumo.merge(estabilidade, on=["alpha", "modelo"])

    holdouts = []
    for alpha in ALPHAS:
        print(f"\nHoldout por ID, alpha={alpha:.2f}")
        metricas_holdout = avaliar_holdout_id(df, alpha)
        holdouts.extend(
            {
                "alpha": alpha,
                "modelo": modelo,
                "rmspe_bloco_id": valor,
            }
            for modelo, valor in metricas_holdout.items()
        )
    resumo = resumo.merge(
        pd.DataFrame(holdouts),
        on=["alpha", "modelo"],
    ).sort_values(["modelo", "rmspe_medio_15_folds"])

    pasta_resultados = os.path.join(raiz, "resultados")
    os.makedirs(pasta_resultados, exist_ok=True)
    detalhado.to_csv(
        os.path.join(
            pasta_resultados,
            "alphas_finalistas_repeated_kfold_detalhado.csv",
        ),
        index=False,
    )
    por_semente.to_csv(
        os.path.join(
            pasta_resultados,
            "alphas_finalistas_resultados_por_semente.csv",
        ),
        index=False,
    )
    resumo.to_csv(
        os.path.join(
            pasta_resultados,
            "alphas_finalistas_repeated_kfold_resumo.csv",
        ),
        index=False,
    )

    colunas = [
        "alpha",
        "modelo",
        "rmspe_medio_15_folds",
        "desvio_15_folds",
        "pior_media_semente",
        "amplitude_entre_sementes",
        "rmspe_bloco_id",
    ]
    print("\nResumo final:")
    print(resumo[colunas].to_string(index=False))


if __name__ == "__main__":
    main()
