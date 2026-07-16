"""Compara CatBoost isolado com diferentes pesos de correspondencia.

Testa alpha 1 e 1,5 com pesos de correspondencia 0%, 15% e 30%.
As previsoes brutas out-of-fold sao salvas para experimentos posteriores.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from testar_catboost_alphas import (
    calcular_rmspe,
    carregar_treino,
    treinar_e_prever,
)


ALPHAS = [1.0, 1.5]
PESOS_CORRESPONDENCIA = [0.0, 0.15, 0.30]
COLUNAS_CORRESPONDENCIA = [
    "tipo",
    "bairro",
    "area_util",
    "area_extra",
    "quartos",
    "suites",
    "vagas",
]


def prever_correspondencias(df_treino, df_validacao):
    tabela = (
        df_treino.groupby(COLUNAS_CORRESPONDENCIA, dropna=False)["preco"]
        .median()
        .rename("preco_correspondencia")
        .reset_index()
    )
    validacao_ordenada = df_validacao.reset_index(drop=True).reset_index(
        names="ordem_original"
    )
    return (
        validacao_ordenada.merge(
            tabela,
            on=COLUNAS_CORRESPONDENCIA,
            how="left",
        )
        .sort_values("ordem_original")["preco_correspondencia"]
        .to_numpy()
    )


def aplicar_correspondencias(
    previsoes_modelo,
    previsoes_correspondencias,
    peso_correspondencia,
):
    previsoes = previsoes_modelo.copy()
    mascara = ~np.isnan(previsoes_correspondencias)
    previsoes[mascara] = (
        (1.0 - peso_correspondencia) * previsoes_modelo[mascara]
        + peso_correspondencia * previsoes_correspondencias[mascara]
    )
    return previsoes


def preparar_divisoes(df):
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(df))
    numero_fold_oof = np.zeros(len(df), dtype=int)
    correspondencias_oof = np.full(len(df), np.nan)
    divisoes = []

    for numero_fold, (indices_treino, indices_validacao) in enumerate(
        folds,
        start=1,
    ):
        df_treino = df.iloc[indices_treino]
        df_validacao = df.iloc[indices_validacao]
        correspondencias = prever_correspondencias(df_treino, df_validacao)
        numero_fold_oof[indices_validacao] = numero_fold
        correspondencias_oof[indices_validacao] = correspondencias
        divisoes.append(
            (
                numero_fold,
                indices_treino,
                indices_validacao,
                correspondencias,
            )
        )

    assert (numero_fold_oof > 0).all()
    return divisoes, numero_fold_oof, correspondencias_oof


def preparar_holdout_id(df):
    quantidade_bloco = round(len(df) * 0.2)
    ids_bloco = set(df.nsmallest(quantidade_bloco, "Id")["Id"])
    df_treino = df[~df["Id"].isin(ids_bloco)]
    df_validacao = df[df["Id"].isin(ids_bloco)]
    correspondencias = prever_correspondencias(df_treino, df_validacao)
    return df_treino, df_validacao, correspondencias


def nome_alpha(alpha):
    return f"alpha_{alpha:.2f}"


def executar_experimento(iteracoes):
    raiz, df = carregar_treino()
    divisoes, numero_fold_oof, correspondencias_oof = preparar_divisoes(df)
    treino_holdout, validacao_holdout, correspondencias_holdout = (
        preparar_holdout_id(df)
    )
    cobertura_oof = np.mean(~np.isnan(correspondencias_oof))
    cobertura_holdout = np.mean(~np.isnan(correspondencias_holdout))

    previsoes_oof = {}
    previsoes_holdout = {}
    duracoes_cv = {}
    resultados_folds = []
    resultados_holdout = []

    for alpha in ALPHAS:
        print(f"\nTreinando CatBoost com alpha={alpha:.2f}")
        previsoes_alpha = np.full(len(df), np.nan)
        duracoes_cv[alpha] = 0.0

        for (
            numero_fold,
            indices_treino,
            indices_validacao,
            correspondencias_fold,
        ) in divisoes:
            df_treino = df.iloc[indices_treino]
            df_validacao = df.iloc[indices_validacao]
            previsoes_brutas, _, duracao = treinar_e_prever(
                df_treino,
                df_validacao,
                alpha,
                iteracoes,
            )
            previsoes_alpha[indices_validacao] = previsoes_brutas
            duracoes_cv[alpha] += duracao

            for peso in PESOS_CORRESPONDENCIA:
                previsoes = aplicar_correspondencias(
                    previsoes_brutas,
                    correspondencias_fold,
                    peso,
                )
                resultados_folds.append(
                    {
                        "alpha": alpha,
                        "peso_correspondencia": peso,
                        "fold": numero_fold,
                        "rmspe": calcular_rmspe(
                            df_validacao["preco"],
                            previsoes,
                        ),
                        "quantidade_correspondencias": int(
                            np.sum(~np.isnan(correspondencias_fold))
                        ),
                        "cobertura_correspondencias": np.mean(
                            ~np.isnan(correspondencias_fold)
                        ),
                    }
                )

            metricas_fold = [
                linha["rmspe"]
                for linha in resultados_folds
                if linha["alpha"] == alpha and linha["fold"] == numero_fold
            ]
            print(
                f"  fold {numero_fold}/5 - "
                + " | ".join(
                    f"corr={peso:.0%}: {rmspe * 100:.2f}%"
                    for peso, rmspe in zip(PESOS_CORRESPONDENCIA, metricas_fold)
                )
            )

        assert np.isfinite(previsoes_alpha).all()
        previsoes_oof[alpha] = previsoes_alpha

        previsoes_brutas_holdout, _, _ = treinar_e_prever(
            treino_holdout,
            validacao_holdout,
            alpha,
            iteracoes,
        )
        previsoes_holdout[alpha] = previsoes_brutas_holdout
        for peso in PESOS_CORRESPONDENCIA:
            previsoes = aplicar_correspondencias(
                previsoes_brutas_holdout,
                correspondencias_holdout,
                peso,
            )
            resultados_holdout.append(
                {
                    "alpha": alpha,
                    "peso_correspondencia": peso,
                    "rmspe_bloco_id": calcular_rmspe(
                        validacao_holdout["preco"],
                        previsoes,
                    ),
                    "quantidade_correspondencias": int(
                        np.sum(~np.isnan(correspondencias_holdout))
                    ),
                    "cobertura_correspondencias": cobertura_holdout,
                }
            )

    detalhado_df = pd.DataFrame(resultados_folds)
    holdout_df = pd.DataFrame(resultados_holdout)
    resumo_df = (
        detalhado_df.groupby(
            ["alpha", "peso_correspondencia"],
            as_index=False,
        )
        .agg(
            rmspe_kfold=("rmspe", "mean"),
            desvio_kfold=("rmspe", "std"),
        )
        .merge(
            holdout_df[
                ["alpha", "peso_correspondencia", "rmspe_bloco_id"]
            ],
            on=["alpha", "peso_correspondencia"],
        )
    )

    rmspe_oof = []
    resultados_faixas = []
    limites_faixas = pd.qcut(df["preco"], q=4, duplicates="drop")
    for alpha in ALPHAS:
        for peso in PESOS_CORRESPONDENCIA:
            previsoes = aplicar_correspondencias(
                previsoes_oof[alpha],
                correspondencias_oof,
                peso,
            )
            rmspe_oof.append(
                {
                    "alpha": alpha,
                    "peso_correspondencia": peso,
                    "rmspe_oof": calcular_rmspe(df["preco"], previsoes),
                }
            )
            for faixa in limites_faixas.cat.categories:
                mascara = limites_faixas == faixa
                resultados_faixas.append(
                    {
                        "alpha": alpha,
                        "peso_correspondencia": peso,
                        "faixa_preco": str(faixa),
                        "quantidade": int(mascara.sum()),
                        "rmspe": calcular_rmspe(
                            df.loc[mascara, "preco"],
                            previsoes[mascara.to_numpy()],
                        ),
                    }
                )

    resumo_df = resumo_df.merge(
        pd.DataFrame(rmspe_oof),
        on=["alpha", "peso_correspondencia"],
    )
    resumo_df["cobertura_oof"] = cobertura_oof
    resumo_df["cobertura_bloco_id"] = cobertura_holdout
    resumo_df["duracao_cv_segundos"] = resumo_df["alpha"].map(duracoes_cv)
    resumo_df["ganho_kfold_vs_corr_0"] = resumo_df.groupby("alpha")[
        "rmspe_kfold"
    ].transform(lambda valores: valores.iloc[0] - valores)
    resumo_df["ganho_bloco_vs_corr_0"] = resumo_df.groupby("alpha")[
        "rmspe_bloco_id"
    ].transform(lambda valores: valores.iloc[0] - valores)
    resumo_df = resumo_df.sort_values(["rmspe_kfold", "rmspe_bloco_id"])

    oof_df = df[["Id", "preco"]].copy()
    oof_df["fold"] = numero_fold_oof
    oof_df["previsao_correspondencia"] = correspondencias_oof
    for alpha in ALPHAS:
        oof_df[f"catboost_{nome_alpha(alpha)}"] = previsoes_oof[alpha]

    holdout_previsoes_df = validacao_holdout[["Id", "preco"]].copy()
    holdout_previsoes_df["previsao_correspondencia"] = correspondencias_holdout
    for alpha in ALPHAS:
        holdout_previsoes_df[f"catboost_{nome_alpha(alpha)}"] = (
            previsoes_holdout[alpha]
        )

    pasta_resultados = os.path.join(raiz, "resultados")
    os.makedirs(pasta_resultados, exist_ok=True)
    arquivos = {
        "catboost_correspondencias_resumo.csv": resumo_df,
        "catboost_correspondencias_folds.csv": detalhado_df,
        "catboost_correspondencias_faixas_preco.csv": pd.DataFrame(
            resultados_faixas
        ),
        "catboost_correspondencias_oof.csv": oof_df,
        "catboost_correspondencias_holdout_id.csv": holdout_previsoes_df,
    }
    for nome_arquivo, dataframe in arquivos.items():
        dataframe.to_csv(
            os.path.join(pasta_resultados, nome_arquivo),
            index=False,
        )

    print("\nResumo CatBoost com correspondencias:")
    colunas = [
        "alpha",
        "peso_correspondencia",
        "rmspe_kfold",
        "desvio_kfold",
        "rmspe_bloco_id",
        "ganho_kfold_vs_corr_0",
        "ganho_bloco_vs_corr_0",
    ]
    print(resumo_df[colunas].to_string(index=False))
    print(f"\nCobertura OOF: {cobertura_oof:.2%}")
    print(f"Cobertura no holdout por ID: {cobertura_holdout:.2%}")
    print(f"Resultados: {os.path.join(pasta_resultados, 'catboost_correspondencias_resumo.csv')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteracoes", type=int, default=1000)
    args = parser.parse_args()
    executar_experimento(args.iteracoes)


if __name__ == "__main__":
    main()
