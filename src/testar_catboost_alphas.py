"""Testa CatBoost isolado com diferentes pesos por faixa de preco.

O alvo e treinado em log1p(preco), com categorias nativas do CatBoost e
sample_weight = 1 / preco**alpha. Nao sao aplicadas correspondencias nem
combinacoes com outros modelos.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold


ALPHAS_PADRAO = [0.0, 1.0, 1.5]
COLUNAS_CATEGORICAS = [
    "tipo",
    "bairro",
    "tipo_vendedor",
    "diferenciais",
]
COLUNAS_COMODIDADES = [
    "churrasqueira",
    "estacionamento",
    "piscina",
    "playground",
    "quadra",
    "s_festas",
    "s_jogos",
    "s_ginastica",
    "sauna",
    "vista_mar",
]


def calcular_rmspe(y_real, y_previsto):
    y_real = np.asarray(y_real)
    y_previsto = np.asarray(y_previsto)
    return np.sqrt(np.mean(((y_real - y_previsto) / y_real) ** 2))


def corrigir_dados(df, corrigir_alvo=False):
    df = df.copy()
    df["area_util"] = df["area_util"].astype(float)
    mascara_area = (
        (df["tipo"] == "Apartamento")
        & (df["quartos"] > 0)
        & ((df["area_util"] / df["quartos"]) > 200)
    )
    df.loc[mascara_area, "area_util"] /= 10

    if corrigir_alvo:
        df.loc[df["Id"].isin([2749, 4316]), "preco"] /= 10
        df.loc[df["Id"] == 6383, "vagas"] = 3

    return df


def carregar_treino():
    raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    caminho = os.path.join(
        raiz,
        "data",
        "conjunto_de_treinamento (5).csv",
    )
    df_original = pd.read_csv(caminho)
    df_base = df_original[
        ~df_original["Id"].isin([5910, 2405, 4568, 6004, 6654])
    ].copy()
    df = corrigir_dados(df_base, corrigir_alvo=True).reset_index(drop=True)
    return raiz, df


def criar_features_catboost(df):
    df_modelo = df.copy()
    df_modelo["area_total"] = (
        df_modelo["area_util"] + df_modelo["area_extra"]
    )
    df_modelo["n_comodidades"] = df_modelo[COLUNAS_COMODIDADES].sum(axis=1)
    df_modelo["area_por_quarto"] = (
        df_modelo["area_util"] / df_modelo["quartos"]
    )
    df_modelo["tem_suite"] = (df_modelo["suites"] > 0).astype(int)
    df_modelo["tem_vaga"] = (df_modelo["vagas"] > 0).astype(int)
    df_modelo["tem_area_extra"] = (df_modelo["area_extra"] > 0).astype(int)

    for coluna in COLUNAS_CATEGORICAS:
        df_modelo[coluna] = df_modelo[coluna].fillna("Ausente").astype(str)

    return df_modelo


def calcular_pesos(y_preco, alpha):
    if alpha == 0:
        return None
    pesos = np.power(y_preco.to_numpy(dtype=float), -alpha)
    return pesos / pesos.mean()


def criar_modelo(iteracoes, semente=42):
    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=iteracoes,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5.0,
        random_strength=1.0,
        bootstrap_type="Bayesian",
        bagging_temperature=1.0,
        boosting_type="Ordered",
        random_seed=semente,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )


def separar_xy(df):
    df_modelo = criar_features_catboost(df)
    y_preco = df_modelo["preco"].copy()
    x = df_modelo.drop(columns=["Id", "preco"])
    return x, y_preco


def treinar_e_prever(
    df_treino,
    df_validacao,
    alpha,
    iteracoes,
    semente_modelo=42,
    prever_treino=False,
):
    x_treino, y_treino = separar_xy(df_treino)
    x_validacao, _ = separar_xy(df_validacao)
    pesos = calcular_pesos(y_treino, alpha)
    modelo = criar_modelo(iteracoes, semente=semente_modelo)

    inicio = time.perf_counter()
    modelo.fit(
        x_treino,
        np.log1p(y_treino),
        cat_features=COLUNAS_CATEGORICAS,
        sample_weight=pesos,
    )
    duracao = time.perf_counter() - inicio
    previsoes_validacao = np.expm1(modelo.predict(x_validacao))

    previsoes_treino = None
    if prever_treino:
        previsoes_treino = np.expm1(modelo.predict(x_treino))

    return previsoes_validacao, previsoes_treino, duracao


def avaliar_alpha(df, alpha, folds, iteracoes):
    previsoes_oof = np.full(len(df), np.nan)
    resultados_folds = []

    for numero_fold, (indices_treino, indices_validacao) in enumerate(
        folds,
        start=1,
    ):
        df_treino = df.iloc[indices_treino]
        df_validacao = df.iloc[indices_validacao]
        previsoes, previsoes_treino, duracao = treinar_e_prever(
            df_treino,
            df_validacao,
            alpha,
            iteracoes,
            prever_treino=True,
        )
        previsoes_oof[indices_validacao] = previsoes
        resultados_folds.append(
            {
                "alpha": alpha,
                "fold": numero_fold,
                "rmspe_validacao": calcular_rmspe(
                    df_validacao["preco"],
                    previsoes,
                ),
                "rmspe_treino": calcular_rmspe(
                    df_treino["preco"],
                    previsoes_treino,
                ),
                "duracao_segundos": duracao,
            }
        )
        print(
            f"  alpha={alpha:.2f} - fold {numero_fold}/5 - "
            f"RMSPE={resultados_folds[-1]['rmspe_validacao'] * 100:.2f}%"
        )

    assert np.isfinite(previsoes_oof).all()
    return previsoes_oof, resultados_folds


def avaliar_holdout_id(df, alpha, iteracoes):
    quantidade_bloco = round(len(df) * 0.2)
    ids_bloco = set(df.nsmallest(quantidade_bloco, "Id")["Id"])
    df_treino = df[~df["Id"].isin(ids_bloco)]
    df_validacao = df[df["Id"].isin(ids_bloco)]
    previsoes, _, duracao = treinar_e_prever(
        df_treino,
        df_validacao,
        alpha,
        iteracoes,
    )
    return calcular_rmspe(df_validacao["preco"], previsoes), duracao


def estatisticas_pesos(precos, alpha):
    pesos = calcular_pesos(precos, alpha)
    if pesos is None:
        pesos = np.ones(len(precos))
    return {
        "peso_min": pesos.min(),
        "peso_mediano": np.median(pesos),
        "peso_p99": np.quantile(pesos, 0.99),
        "peso_max": pesos.max(),
    }


def executar_experimento(alphas, iteracoes):
    raiz, df = carregar_treino()
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(df))
    detalhado = []
    resumos = []
    faixas = []
    erros_oof = {}
    limites_faixas = pd.qcut(df["preco"], q=4, duplicates="drop")

    for alpha in alphas:
        print(f"\nAvaliando CatBoost com alpha={alpha:.2f}")
        previsoes_oof, resultados_folds = avaliar_alpha(
            df,
            alpha,
            folds,
            iteracoes,
        )
        rmspe_holdout, duracao_holdout = avaliar_holdout_id(
            df,
            alpha,
            iteracoes,
        )
        detalhado.extend(resultados_folds)
        metricas_folds = [linha["rmspe_validacao"] for linha in resultados_folds]
        metricas_treino = [linha["rmspe_treino"] for linha in resultados_folds]
        resumos.append(
            {
                "alpha": alpha,
                "iteracoes": iteracoes,
                "rmspe_kfold": np.mean(metricas_folds),
                "desvio_kfold": np.std(metricas_folds),
                "rmspe_oof": calcular_rmspe(df["preco"], previsoes_oof),
                "rmspe_treino": np.mean(metricas_treino),
                "rmspe_bloco_id": rmspe_holdout,
                "duracao_cv_segundos": sum(
                    linha["duracao_segundos"] for linha in resultados_folds
                ),
                "duracao_holdout_segundos": duracao_holdout,
                **estatisticas_pesos(df["preco"], alpha),
            }
        )

        for faixa in limites_faixas.cat.categories:
            mascara = limites_faixas == faixa
            faixas.append(
                {
                    "alpha": alpha,
                    "faixa_preco": str(faixa),
                    "quantidade": int(mascara.sum()),
                    "rmspe": calcular_rmspe(
                        df.loc[mascara, "preco"],
                        previsoes_oof[mascara.to_numpy()],
                    ),
                }
            )

        erros_oof[f"alpha_{alpha:.2f}"] = (
            previsoes_oof - df["preco"].to_numpy()
        ) / df["preco"].to_numpy()

    resumo_df = pd.DataFrame(resumos).sort_values(
        ["rmspe_kfold", "rmspe_bloco_id"]
    )
    detalhado_df = pd.DataFrame(detalhado)
    faixas_df = pd.DataFrame(faixas)
    correlacoes_df = pd.DataFrame(erros_oof).corr()

    pasta_resultados = os.path.join(raiz, "resultados")
    os.makedirs(pasta_resultados, exist_ok=True)
    caminho_resumo = os.path.join(
        pasta_resultados,
        "catboost_alphas_resumo.csv",
    )
    caminho_detalhado = os.path.join(
        pasta_resultados,
        "catboost_alphas_folds.csv",
    )
    caminho_faixas = os.path.join(
        pasta_resultados,
        "catboost_alphas_faixas_preco.csv",
    )
    caminho_correlacoes = os.path.join(
        pasta_resultados,
        "catboost_alphas_correlacao_erros.csv",
    )
    resumo_df.to_csv(caminho_resumo, index=False)
    detalhado_df.to_csv(caminho_detalhado, index=False)
    faixas_df.to_csv(caminho_faixas, index=False)
    correlacoes_df.to_csv(caminho_correlacoes)

    print("\nResumo CatBoost isolado:")
    print(
        resumo_df[
            [
                "alpha",
                "rmspe_kfold",
                "desvio_kfold",
                "rmspe_treino",
                "rmspe_bloco_id",
                "duracao_cv_segundos",
            ]
        ].to_string(index=False)
    )
    print(f"\nResultados: {caminho_resumo}")
    print(f"Folds: {caminho_detalhado}")
    print(f"Faixas de preco: {caminho_faixas}")
    print(f"Correlacoes: {caminho_correlacoes}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=ALPHAS_PADRAO,
    )
    parser.add_argument("--iteracoes", type=int, default=1000)
    args = parser.parse_args()
    executar_experimento(args.alphas, args.iteracoes)


if __name__ == "__main__":
    main()
