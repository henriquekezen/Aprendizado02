"""Compara juizes cross-fit para misturar generalista e especialista."""

import json
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from testar_catboost_alphas import (
    COLUNAS_CATEGORICAS,
    carregar_treino,
    criar_features_catboost,
)


LIMITE_BARATO_ORACULO = 328_600.0


def rmspe(y, p):
    return np.sqrt(np.mean(np.square((p - y) / y)))


def preparar_features_com_previsoes(
    df,
    geral,
    especialista,
    cat_global,
    blend_arvores,
    tem_correspondencia,
):
    x = criar_features_catboost(df).drop(columns=["Id", "preco"])
    geral = np.asarray(geral)
    especialista = np.asarray(especialista)
    x["pred_generalista"] = geral
    x["pred_especialista"] = especialista
    x["log_pred_generalista"] = np.log1p(geral)
    x["log_pred_especialista"] = np.log1p(especialista)
    x["diferenca_predicoes"] = especialista - geral
    x["razao_especialista_generalista"] = especialista / geral
    x["pred_cat_global"] = np.asarray(cat_global)
    x["pred_blend_arvores"] = np.asarray(blend_arvores)
    x["discordancia_componentes"] = (
        x["pred_cat_global"] - x["pred_blend_arvores"]
    ) / geral
    x["tem_correspondencia"] = np.asarray(tem_correspondencia, dtype=int)
    assert np.isfinite(x.select_dtypes(include=[np.number])).all().all()
    return x


def preparar_features(df, geral_oof, especialista_oof, cat_oof):
    return preparar_features_com_previsoes(
        df=df,
        geral=geral_oof["generalista_cat60_arvores40"],
        especialista=especialista_oof["iter_600_calibrado_cf"],
        cat_global=geral_oof["catboost_alpha150_corr30"],
        blend_arvores=geral_oof["blend_arvores_alpha150_corr30"],
        tem_correspondencia=cat_oof["previsao_correspondencia"].notna(),
    )


def parametros_comuns():
    return {
        "iterations": 500,
        "learning_rate": 0.03,
        "depth": 5,
        "l2_leaf_reg": 5.0,
        "random_strength": 0.5,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 1.0,
        "boosting_type": "Ordered",
        "random_seed": 42,
        "allow_writing_files": False,
        "verbose": False,
        "thread_count": -1,
    }


def criar_classificador():
    return CatBoostClassifier(loss_function="Logloss", **parametros_comuns())


def criar_regressor():
    return CatBoostRegressor(loss_function="RMSE", **parametros_comuns())


def treinar_scores_crossfit(x, df_meta):
    scores = {
        "preco_generalista_invertido": -df_meta["generalista"].to_numpy(),
        "classificador_barato": np.full(len(df_meta), np.nan),
        "classificador_barato_utilidade": np.full(len(df_meta), np.nan),
        "classificador_utilidade": np.full(len(df_meta), np.nan),
        "regressor_delta": np.full(len(df_meta), np.nan),
    }
    alvo_barato = (df_meta["preco"] <= LIMITE_BARATO_ORACULO).astype(int)
    alvo_utilidade = (df_meta["delta_loss"] > 0).astype(int)
    peso_utilidade = np.abs(df_meta["delta_loss"].to_numpy())
    limite_peso = np.quantile(peso_utilidade, 0.99)
    peso_utilidade = np.clip(peso_utilidade, 1e-5, limite_peso)
    peso_utilidade /= peso_utilidade.mean()
    delta_clip = np.clip(
        df_meta["delta_loss"].to_numpy(),
        np.quantile(df_meta["delta_loss"], 0.01),
        np.quantile(df_meta["delta_loss"], 0.99),
    )

    for fold in sorted(df_meta["fold"].unique()):
        treino = df_meta["fold"].ne(fold).to_numpy()
        validacao = df_meta["fold"].eq(fold).to_numpy()

        modelo = criar_classificador()
        modelo.fit(
            x.loc[treino],
            alvo_barato.loc[treino],
            cat_features=COLUNAS_CATEGORICAS,
        )
        scores["classificador_barato"][validacao] = modelo.predict_proba(
            x.loc[validacao]
        )[:, 1]

        modelo = criar_classificador()
        modelo.fit(
            x.loc[treino],
            alvo_barato.loc[treino],
            cat_features=COLUNAS_CATEGORICAS,
            sample_weight=peso_utilidade[treino],
        )
        scores["classificador_barato_utilidade"][validacao] = (
            modelo.predict_proba(x.loc[validacao])[:, 1]
        )

        modelo = criar_classificador()
        modelo.fit(
            x.loc[treino],
            alvo_utilidade.loc[treino],
            cat_features=COLUNAS_CATEGORICAS,
            sample_weight=peso_utilidade[treino],
        )
        scores["classificador_utilidade"][validacao] = modelo.predict_proba(
            x.loc[validacao]
        )[:, 1]

        modelo_reg = criar_regressor()
        modelo_reg.fit(
            x.loc[treino],
            delta_clip[treino],
            cat_features=COLUNAS_CATEGORICAS,
            sample_weight=peso_utilidade[treino],
        )
        scores["regressor_delta"][validacao] = modelo_reg.predict(
            x.loc[validacao]
        )
        print(f"Juizes do fold {fold}/5 concluidos")

    assert all(np.isfinite(valores).all() for valores in scores.values())
    return scores


def pesos_sigmoide(score, centro, escala):
    z = np.clip((score - centro) / escala, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def grade_mapeamento(scores_treino):
    quantis = np.linspace(0.35, 0.95, 13)
    centros = np.unique(np.quantile(scores_treino, quantis))
    desvio = max(float(np.std(scores_treino)), 1e-9)
    escalas = desvio * np.array([0.05, 0.10, 0.20, 0.35, 0.50, 0.75])
    return [(centro, escala) for centro in centros for escala in escalas]


def escolher_mapeamento(y, geral, especialista, score, mascara_treino):
    melhor = None
    for centro, escala in grade_mapeamento(score[mascara_treino]):
        pesos = pesos_sigmoide(score[mascara_treino], centro, escala)
        pred = geral[mascara_treino] + pesos * (
            especialista[mascara_treino] - geral[mascara_treino]
        )
        erro = rmspe(y[mascara_treino], pred)
        if melhor is None or erro < melhor["rmspe"]:
            melhor = {"centro": centro, "escala": escala, "rmspe": erro}
    return melhor


def mapear_scores_crossfit(df_meta, scores):
    y = df_meta["preco"].to_numpy()
    geral = df_meta["generalista"].to_numpy()
    especialista = df_meta["especialista"].to_numpy()
    resultados = []
    saida = {}
    parametros_folds = []

    for nome, score in scores.items():
        pesos_cf = np.full(len(df_meta), np.nan)
        for fold in sorted(df_meta["fold"].unique()):
            treino = df_meta["fold"].ne(fold).to_numpy()
            validacao = df_meta["fold"].eq(fold).to_numpy()
            melhor = escolher_mapeamento(
                y, geral, especialista, score, treino
            )
            pesos_cf[validacao] = pesos_sigmoide(
                score[validacao], melhor["centro"], melhor["escala"]
            )
            parametros_folds.append(
                {
                    "juiz": nome,
                    "fold": int(fold),
                    **melhor,
                }
            )
        pred_cf = geral + pesos_cf * (especialista - geral)
        melhor_final = escolher_mapeamento(
            y,
            geral,
            especialista,
            score,
            np.ones(len(df_meta), dtype=bool),
        )
        saida[nome] = {
            "score": score,
            "peso_cf": pesos_cf,
            "pred_cf": pred_cf,
            "parametro_final": melhor_final,
        }
        fold_scores = []
        for fold in sorted(df_meta["fold"].unique()):
            mascara = df_meta["fold"].eq(fold).to_numpy()
            fold_scores.append(rmspe(y[mascara], pred_cf[mascara]))
        resultados.append(
            {
                "juiz": nome,
                "rmspe_crossfit": rmspe(y, pred_cf),
                "media_peso": pesos_cf.mean(),
                "mediana_peso": np.median(pesos_cf),
                "p90_peso": np.quantile(pesos_cf, 0.90),
                "fracao_peso_maior_05": np.mean(pesos_cf > 0.5),
                "centro_final": melhor_final["centro"],
                "escala_final": melhor_final["escala"],
                **{
                    f"rmspe_fold_{i}": valor
                    for i, valor in enumerate(fold_scores, start=1)
                },
            }
        )
    return saida, pd.DataFrame(resultados), pd.DataFrame(parametros_folds)


def main():
    raiz, df = carregar_treino()
    pasta = os.path.join(raiz, "resultados")
    geral = pd.read_csv(os.path.join(pasta, "generalista_60_40_oof.csv"))
    especialista = pd.read_csv(
        os.path.join(
            pasta, "catboost_baratos_curva_iteracoes_d7_rs0_oof.csv"
        )
    )
    cat_oof = pd.read_csv(
        os.path.join(pasta, "catboost_correspondencias_oof.csv")
    )
    assert np.array_equal(df["Id"].to_numpy(), geral["Id"].to_numpy())
    assert np.array_equal(df["Id"].to_numpy(), especialista["Id"].to_numpy())

    meta = df[["Id", "preco"]].copy()
    meta["fold"] = geral["fold"]
    meta["generalista"] = geral["generalista_cat60_arvores40"]
    meta["especialista"] = especialista["iter_600_calibrado_cf"]
    meta["loss_generalista"] = np.square(
        (meta["generalista"] - meta["preco"]) / meta["preco"]
    )
    meta["loss_especialista"] = np.square(
        (meta["especialista"] - meta["preco"]) / meta["preco"]
    )
    meta["delta_loss"] = meta["loss_generalista"] - meta["loss_especialista"]
    x = preparar_features(df, geral, especialista, cat_oof)
    scores = treinar_scores_crossfit(x, meta)
    saida, resumo, parametros = mapear_scores_crossfit(meta, scores)

    for nome, dados in saida.items():
        meta[f"score_{nome}"] = dados["score"]
        meta[f"peso_{nome}"] = dados["peso_cf"]
        meta[f"pred_{nome}"] = dados["pred_cf"]
    resumo = resumo.sort_values("rmspe_crossfit")
    resumo.to_csv(os.path.join(pasta, "juizes_especialista_resumo.csv"), index=False)
    parametros.to_csv(
        os.path.join(pasta, "juizes_especialista_parametros_folds.csv"),
        index=False,
    )
    meta.to_csv(os.path.join(pasta, "juizes_especialista_oof.csv"), index=False)
    parametros_finais = {
        nome: {
            "centro": float(dados["parametro_final"]["centro"]),
            "escala": float(dados["parametro_final"]["escala"]),
        }
        for nome, dados in saida.items()
    }
    with open(
        os.path.join(pasta, "juizes_especialista_parametros_finais.json"),
        "w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(parametros_finais, arquivo, ensure_ascii=False, indent=2)

    print("\nResumo dos juizes:")
    print(resumo.to_string(index=False))


if __name__ == "__main__":
    main()
