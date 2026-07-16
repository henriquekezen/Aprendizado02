"""Reconstrui o OOF do generalista publico de 21,45%.

Generalista = 60% CatBoost alpha 1,5 + 40% blend 50/50 de XGBoost e
LightGBM alpha 1,5. Todos os componentes usam correspondencia de 30%.
"""

import os
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from pre_processamento import (
    criar_features_modelo_bairro_categorico,
    selecionar_bairros_frequentes,
)
from testar_catboost_alphas import carregar_treino
from testar_catboost_correspondencias import (
    aplicar_correspondencias,
    prever_correspondencias,
)


ALPHA = 1.5
PESO_CORRESPONDENCIA = 0.30


def calcular_rmspe(y_real, y_previsto):
    y_real = np.asarray(y_real)
    y_previsto = np.asarray(y_previsto)
    return np.sqrt(np.mean(((y_previsto - y_real) / y_real) ** 2))


def calcular_pesos(y_preco):
    pesos = np.power(y_preco.to_numpy(dtype=float), -ALPHA)
    return pesos / pesos.mean()


def preparar_features(df_treino, df_validacao):
    bairros = selecionar_bairros_frequentes(df_treino, minimo_imoveis=10)
    treino = criar_features_modelo_bairro_categorico(df_treino, bairros)
    validacao = criar_features_modelo_bairro_categorico(df_validacao, bairros)
    return treino, validacao


def criar_xgboost():
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=800,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=4,
        random_state=42,
        n_jobs=-1,
    )


def criar_lightgbm():
    return LGBMRegressor(
        objective="regression",
        n_estimators=400,
        learning_rate=0.06,
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


def treinar_arvores(df_treino, df_validacao):
    treino, validacao = preparar_features(df_treino, df_validacao)
    x_treino = treino.drop(columns=["Id", "preco"])
    y_treino = treino["preco"]
    x_validacao = validacao.drop(columns=["Id", "preco"], errors="ignore")
    pesos = calcular_pesos(y_treino)

    inicio = time.perf_counter()
    xgb = criar_xgboost()
    xgb.fit(x_treino, np.log1p(y_treino), sample_weight=pesos)
    previsoes_xgb = np.expm1(xgb.predict(x_validacao))

    lgb = criar_lightgbm()
    lgb.fit(
        x_treino,
        np.log1p(y_treino),
        sample_weight=pesos,
        categorical_feature=["bairro"],
    )
    previsoes_lgb = np.expm1(lgb.predict(x_validacao))
    duracao = time.perf_counter() - inicio
    return previsoes_xgb, previsoes_lgb, duracao


def carregar_catboost_oof(raiz, df):
    caminho = os.path.join(
        raiz,
        "resultados",
        "catboost_correspondencias_oof.csv",
    )
    cat = pd.read_csv(caminho)
    assert np.array_equal(cat["Id"].to_numpy(), df["Id"].to_numpy())
    previsoes = aplicar_correspondencias(
        cat["catboost_alpha_1.50"].to_numpy(),
        cat["previsao_correspondencia"].to_numpy(),
        PESO_CORRESPONDENCIA,
    )
    return cat, previsoes


def main():
    raiz, df = carregar_treino()
    splits = list(KFold(n_splits=5, shuffle=True, random_state=42).split(df))
    cat_oof, previsoes_cat = carregar_catboost_oof(raiz, df)
    previsoes_xgb = np.full(len(df), np.nan)
    previsoes_lgb = np.full(len(df), np.nan)
    previsoes_arvores = np.full(len(df), np.nan)
    numero_fold = np.zeros(len(df), dtype=int)
    resultados_folds = []

    for fold, (indices_treino, indices_validacao) in enumerate(splits, start=1):
        treino = df.iloc[indices_treino]
        validacao = df.iloc[indices_validacao]
        pred_xgb, pred_lgb, duracao = treinar_arvores(treino, validacao)
        pred_corr = prever_correspondencias(treino, validacao)
        pred_arvores_bruto = 0.50 * pred_xgb + 0.50 * pred_lgb
        pred_arvores = aplicar_correspondencias(
            pred_arvores_bruto,
            pred_corr,
            PESO_CORRESPONDENCIA,
        )
        previsoes_xgb[indices_validacao] = pred_xgb
        previsoes_lgb[indices_validacao] = pred_lgb
        previsoes_arvores[indices_validacao] = pred_arvores
        numero_fold[indices_validacao] = fold
        pred_geral = 0.60 * previsoes_cat[indices_validacao] + 0.40 * pred_arvores
        resultados_folds.append(
            {
                "fold": fold,
                "rmspe_catboost": calcular_rmspe(
                    validacao["preco"], previsoes_cat[indices_validacao]
                ),
                "rmspe_blend_arvores": calcular_rmspe(
                    validacao["preco"], pred_arvores
                ),
                "rmspe_generalista": calcular_rmspe(
                    validacao["preco"], pred_geral
                ),
                "duracao_arvores_segundos": duracao,
            }
        )
        print(
            f"Fold {fold}/5 - generalista: "
            f"{resultados_folds[-1]['rmspe_generalista'] * 100:.2f}%"
        )

    assert np.isfinite(previsoes_arvores).all()
    assert np.array_equal(numero_fold, cat_oof["fold"].to_numpy())
    previsoes_generalista = 0.60 * previsoes_cat + 0.40 * previsoes_arvores
    oof = pd.DataFrame(
        {
            "Id": df["Id"],
            "preco": df["preco"],
            "fold": numero_fold,
            "xgboost_alpha150_bruto": previsoes_xgb,
            "lightgbm_alpha150_bruto": previsoes_lgb,
            "blend_arvores_alpha150_corr30": previsoes_arvores,
            "catboost_alpha150_corr30": previsoes_cat,
            "generalista_cat60_arvores40": previsoes_generalista,
        }
    )
    pasta = os.path.join(raiz, "resultados")
    caminho_oof = os.path.join(pasta, "generalista_60_40_oof.csv")
    caminho_folds = os.path.join(pasta, "generalista_60_40_folds.csv")
    oof.to_csv(caminho_oof, index=False)
    pd.DataFrame(resultados_folds).to_csv(caminho_folds, index=False)

    print("\nResumo:")
    print(pd.DataFrame(resultados_folds).to_string(index=False))
    print(
        "RMSPE OOF generalista: "
        f"{calcular_rmspe(df['preco'], previsoes_generalista) * 100:.2f}%"
    )
    print(f"OOF: {caminho_oof}")


if __name__ == "__main__":
    main()
