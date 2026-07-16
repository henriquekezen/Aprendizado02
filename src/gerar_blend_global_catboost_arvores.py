"""Gera blend global: 60% CatBoost e 40% blend XGBoost/LightGBM."""

import os

import numpy as np
import pandas as pd


PESO_CATBOOST = 0.60
PESO_BLEND_ARVORES = 0.40
ARQUIVO_CATBOOST = "submission_catboost_a150_corr30.csv"
ARQUIVO_BLEND_ARVORES = (
    "submission_blend_50_sample_weight_a150_corr30.csv"
)
ARQUIVO_SAIDA = (
    "submission_blend_global_cat60_xgb20_lgb20_a150_corr30.csv"
)


def main():
    raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pasta_submissions = os.path.join(raiz, "submissions")
    catboost = pd.read_csv(os.path.join(pasta_submissions, ARQUIVO_CATBOOST))
    blend_arvores = pd.read_csv(
        os.path.join(pasta_submissions, ARQUIVO_BLEND_ARVORES)
    )

    assert list(catboost.columns) == ["Id", "preco"]
    assert list(blend_arvores.columns) == ["Id", "preco"]
    assert catboost.shape == blend_arvores.shape == (2000, 2)
    assert np.array_equal(
        catboost["Id"].to_numpy(),
        blend_arvores["Id"].to_numpy(),
    )

    submission = catboost[["Id"]].copy()
    submission["preco"] = (
        PESO_CATBOOST * catboost["preco"]
        + PESO_BLEND_ARVORES * blend_arvores["preco"]
    ).round(2)
    caminho_saida = os.path.join(pasta_submissions, ARQUIVO_SAIDA)
    submission.to_csv(caminho_saida, index=False)

    assert submission["Id"].is_unique
    assert not submission.isna().any().any()
    assert (submission["preco"] > 0).all()
    print(f"Submission: {caminho_saida}")


if __name__ == "__main__":
    main()
