"""Avalia intensidades de sample weight para aproximar melhor o RMSPE.

O alvo continua em log1p(preco). Os pesos seguem 1 / preco**alpha:
alpha=0 reproduz o treino sem pesos e valores maiores enfatizam
progressivamente os imoveis baratos.

Uso:
    python src/testar_alphas_sample_weights.py avaliar
    python src/testar_alphas_sample_weights.py gerar --alphas 0.75 1.0
"""

import argparse
import os

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from pre_processamento import (
    criar_features_modelo_bairro_categorico,
    selecionar_bairros_frequentes,
)


ALPHAS_PADRAO = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50]
MODELOS = ["xgboost", "lightgbm", "blend_50"]
COLUNAS_CORRESPONDENCIA = [
    "tipo",
    "bairro",
    "area_util",
    "area_extra",
    "quartos",
    "suites",
    "vagas",
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


def carregar_dados():
    raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    treino_original = pd.read_csv(
        os.path.join(raiz, "data", "conjunto_de_treinamento (5).csv")
    )
    teste_original = pd.read_csv(
        os.path.join(raiz, "data", "conjunto_de_teste (3).csv")
    )
    treino_base = treino_original[
        ~treino_original["Id"].isin([5910, 2405, 4568, 6004, 6654])
    ].copy()
    treino = corrigir_dados(treino_base, corrigir_alvo=True).reset_index(
        drop=True
    )
    teste = corrigir_dados(teste_original).reset_index(drop=True)
    return raiz, treino, teste


def preparar_features(df_treino, df_validacao):
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
    return treino_modelo, validacao_modelo


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


def calcular_pesos(y_preco, alpha):
    if alpha == 0:
        return None
    pesos = np.power(y_preco.to_numpy(dtype=float), -alpha)
    return pesos / pesos.mean()


def prever_modelos(df_treino, df_validacao, alpha):
    treino_modelo, validacao_modelo = preparar_features(
        df_treino,
        df_validacao,
    )
    x_treino = treino_modelo.drop(columns=["Id", "preco"])
    y_treino_log = np.log1p(treino_modelo["preco"])
    x_validacao = validacao_modelo.drop(
        columns=["Id", "preco"],
        errors="ignore",
    )
    pesos = calcular_pesos(treino_modelo["preco"], alpha)

    modelo_xgb = criar_xgboost()
    modelo_xgb.fit(x_treino, y_treino_log, sample_weight=pesos)
    previsoes_xgb = np.expm1(modelo_xgb.predict(x_validacao))

    modelo_lgbm = criar_lightgbm()
    modelo_lgbm.fit(
        x_treino,
        y_treino_log,
        sample_weight=pesos,
        categorical_feature=["bairro"],
    )
    previsoes_lgbm = np.expm1(modelo_lgbm.predict(x_validacao))
    return previsoes_xgb, previsoes_lgbm


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


def aplicar_correspondencias(previsoes_modelo, previsoes_correspondencias):
    previsoes = previsoes_modelo.copy()
    mascara = ~np.isnan(previsoes_correspondencias)
    previsoes[mascara] = (
        0.70 * previsoes_modelo[mascara]
        + 0.30 * previsoes_correspondencias[mascara]
    )
    return previsoes


def calcular_todas_previsoes(
    previsoes_xgb,
    previsoes_lgbm,
    previsoes_correspondencias,
):
    previsoes_blend = 0.50 * previsoes_xgb + 0.50 * previsoes_lgbm
    return {
        "xgboost": aplicar_correspondencias(
            previsoes_xgb,
            previsoes_correspondencias,
        ),
        "lightgbm": aplicar_correspondencias(
            previsoes_lgbm,
            previsoes_correspondencias,
        ),
        "blend_50": aplicar_correspondencias(
            previsoes_blend,
            previsoes_correspondencias,
        ),
    }


def avaliar_alpha(df, alpha, folds):
    previsoes_oof = {nome: np.full(len(df), np.nan) for nome in MODELOS}
    metricas_folds = {nome: [] for nome in MODELOS}

    for numero_fold, (indices_treino, indices_validacao) in enumerate(
        folds,
        start=1,
    ):
        df_treino = df.iloc[indices_treino]
        df_validacao = df.iloc[indices_validacao]
        previsoes_xgb, previsoes_lgbm = prever_modelos(
            df_treino,
            df_validacao,
            alpha,
        )
        previsoes_correspondencias = prever_correspondencias(
            df_treino,
            df_validacao,
        )
        previsoes = calcular_todas_previsoes(
            previsoes_xgb,
            previsoes_lgbm,
            previsoes_correspondencias,
        )
        for nome, valores in previsoes.items():
            previsoes_oof[nome][indices_validacao] = valores
            metricas_folds[nome].append(
                calcular_rmspe(df_validacao["preco"], valores)
            )
        print(f"  alpha={alpha:.2f} - fold {numero_fold}/5")

    assert all(np.isfinite(valores).all() for valores in previsoes_oof.values())
    return previsoes_oof, metricas_folds


def avaliar_holdout_id(df, alpha):
    quantidade_bloco = round(len(df) * 0.2)
    ids_bloco = set(df.nsmallest(quantidade_bloco, "Id")["Id"])
    df_treino = df[~df["Id"].isin(ids_bloco)]
    df_validacao = df[df["Id"].isin(ids_bloco)]
    previsoes_xgb, previsoes_lgbm = prever_modelos(
        df_treino,
        df_validacao,
        alpha,
    )
    previsoes_correspondencias = prever_correspondencias(df_treino, df_validacao)
    previsoes = calcular_todas_previsoes(
        previsoes_xgb,
        previsoes_lgbm,
        previsoes_correspondencias,
    )
    return {
        nome: calcular_rmspe(df_validacao["preco"], valores)
        for nome, valores in previsoes.items()
    }


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


def avaliar_grade(alphas):
    raiz, df, _ = carregar_dados()
    pasta_resultados = os.path.join(raiz, "resultados")
    os.makedirs(pasta_resultados, exist_ok=True)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(df))
    resultados = []
    faixas = []
    erros_blend = {}
    limites_faixas = pd.qcut(df["preco"], q=4, duplicates="drop")

    for alpha in alphas:
        print(f"\nAvaliando alpha={alpha:.2f}")
        previsoes_oof, metricas_folds = avaliar_alpha(df, alpha, folds)
        metricas_holdout = avaliar_holdout_id(df, alpha)
        estatisticas = estatisticas_pesos(df["preco"], alpha)

        for nome in MODELOS:
            valores_folds = metricas_folds[nome]
            resultados.append(
                {
                    "alpha": alpha,
                    "modelo": nome,
                    "rmspe_kfold": np.mean(valores_folds),
                    "desvio_kfold": np.std(valores_folds),
                    "rmspe_oof": calcular_rmspe(
                        df["preco"],
                        previsoes_oof[nome],
                    ),
                    "rmspe_bloco_id": metricas_holdout[nome],
                    **{
                        f"rmspe_fold_{indice}": valor
                        for indice, valor in enumerate(valores_folds, start=1)
                    },
                    **estatisticas,
                }
            )

        for faixa in limites_faixas.cat.categories:
            mascara = limites_faixas == faixa
            for nome in MODELOS:
                faixas.append(
                    {
                        "alpha": alpha,
                        "modelo": nome,
                        "faixa_preco": str(faixa),
                        "quantidade": int(mascara.sum()),
                        "rmspe": calcular_rmspe(
                            df.loc[mascara, "preco"],
                            previsoes_oof[nome][mascara.to_numpy()],
                        ),
                    }
                )

        erros_blend[f"alpha_{alpha:.2f}"] = (
            previsoes_oof["blend_50"] - df["preco"].to_numpy()
        ) / df["preco"].to_numpy()

    resultados_df = pd.DataFrame(resultados).sort_values(
        ["modelo", "rmspe_kfold", "rmspe_bloco_id"]
    )
    faixas_df = pd.DataFrame(faixas)
    correlacoes_df = pd.DataFrame(erros_blend).corr()

    caminho_resultados = os.path.join(
        pasta_resultados,
        "comparacao_alphas_sample_weights.csv",
    )
    caminho_faixas = os.path.join(
        pasta_resultados,
        "rmspe_alphas_por_faixa_preco.csv",
    )
    caminho_correlacoes = os.path.join(
        pasta_resultados,
        "correlacao_erros_oof_alphas.csv",
    )
    resultados_df.to_csv(caminho_resultados, index=False)
    faixas_df.to_csv(caminho_faixas, index=False)
    correlacoes_df.to_csv(caminho_correlacoes)

    print("\nResumo ordenado por modelo e RMSPE medio:")
    print(
        resultados_df[
            [
                "alpha",
                "modelo",
                "rmspe_kfold",
                "desvio_kfold",
                "rmspe_bloco_id",
                "peso_max",
            ]
        ].to_string(index=False)
    )
    print(f"\nResultados: {caminho_resultados}")
    print(f"Faixas de preco: {caminho_faixas}")
    print(f"Correlacoes: {caminho_correlacoes}")


def sufixo_alpha(alpha):
    return f"a{int(round(alpha * 100)):03d}"


def gerar_submissions(alphas, modelos):
    if len(alphas) * len(modelos) > 4:
        raise ValueError("Foram solicitadas mais de 4 submissions.")

    raiz, df_treino, df_teste = carregar_dados()
    modelo_resposta = pd.read_csv(
        os.path.join(raiz, "data", "exemplo_arquivo_respostas.csv")
    )
    assert np.array_equal(
        modelo_resposta["Id"].to_numpy(),
        df_teste["Id"].to_numpy(),
    )
    pasta_saida = os.path.join(raiz, "submissions")
    os.makedirs(pasta_saida, exist_ok=True)

    for alpha in alphas:
        print(f"Treinando modelos finais com alpha={alpha:.2f}")
        previsoes_xgb, previsoes_lgbm = prever_modelos(
            df_treino,
            df_teste,
            alpha,
        )
        previsoes_correspondencias = prever_correspondencias(
            df_treino,
            df_teste,
        )
        previsoes = calcular_todas_previsoes(
            previsoes_xgb,
            previsoes_lgbm,
            previsoes_correspondencias,
        )

        for nome_modelo in modelos:
            submission = modelo_resposta.copy()
            submission["preco"] = previsoes[nome_modelo].round(2)
            arquivo = (
                f"submission_{nome_modelo}_sample_weight_"
                f"{sufixo_alpha(alpha)}_corr30.csv"
            )
            caminho = os.path.join(pasta_saida, arquivo)
            submission.to_csv(caminho, index=False)

            assert submission.shape == (2000, 2)
            assert submission["Id"].is_unique
            assert not submission.isna().any().any()
            assert (submission["preco"] > 0).all()
            print(f"Submission: {caminho}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="modo", required=True)
    parser_avaliar = subparsers.add_parser("avaliar")
    parser_avaliar.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=ALPHAS_PADRAO,
    )
    parser_gerar = subparsers.add_parser("gerar")
    parser_gerar.add_argument("--alphas", type=float, nargs="+", required=True)
    parser_gerar.add_argument(
        "--modelos",
        choices=MODELOS,
        nargs="+",
        default=["blend_50"],
    )
    args = parser.parse_args()

    if args.modo == "avaliar":
        avaliar_grade(args.alphas)
    else:
        gerar_submissions(args.alphas, args.modelos)


if __name__ == "__main__":
    main()
