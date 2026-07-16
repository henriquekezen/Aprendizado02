"""Gera submissions CatBoost para alpha 1 e 1,5 com correspondencia de 30%."""

import os
import time

import numpy as np
import pandas as pd

from testar_catboost_alphas import (
    COLUNAS_CATEGORICAS,
    calcular_pesos,
    carregar_treino,
    corrigir_dados,
    criar_modelo,
    separar_xy,
)
from testar_catboost_correspondencias import (
    aplicar_correspondencias,
    prever_correspondencias,
)


ALPHAS = [1.0, 1.5]
PESO_CORRESPONDENCIA = 0.30
ITERACOES = 1000


def nome_alpha(alpha):
    return f"a{int(round(alpha * 100)):03d}"


def carregar_teste(raiz):
    caminho = os.path.join(
        raiz,
        "data",
        "conjunto_de_teste (3).csv",
    )
    return corrigir_dados(pd.read_csv(caminho)).reset_index(drop=True)


def treinar_e_prever_teste(df_treino, df_teste, alpha):
    x_treino, y_treino = separar_xy(df_treino)
    x_teste, _ = separar_xy(df_teste.assign(preco=np.nan))
    pesos = calcular_pesos(y_treino, alpha)
    modelo = criar_modelo(ITERACOES, semente=42)

    inicio = time.perf_counter()
    modelo.fit(
        x_treino,
        np.log1p(y_treino),
        cat_features=COLUNAS_CATEGORICAS,
        sample_weight=pesos,
    )
    duracao = time.perf_counter() - inicio
    previsoes = np.expm1(modelo.predict(x_teste))
    return previsoes, duracao


def main():
    raiz, df_treino = carregar_treino()
    df_teste = carregar_teste(raiz)
    modelo_resposta = pd.read_csv(
        os.path.join(raiz, "data", "exemplo_arquivo_respostas.csv")
    )
    assert np.array_equal(
        modelo_resposta["Id"].to_numpy(),
        df_teste["Id"].to_numpy(),
    )

    previsoes_correspondencias = prever_correspondencias(df_treino, df_teste)
    pasta_submissions = os.path.join(raiz, "submissions")
    pasta_resultados = os.path.join(raiz, "resultados")
    os.makedirs(pasta_submissions, exist_ok=True)
    os.makedirs(pasta_resultados, exist_ok=True)

    previsoes_teste = df_teste[["Id"]].copy()
    previsoes_teste["previsao_correspondencia"] = previsoes_correspondencias

    for alpha in ALPHAS:
        print(f"Treinando CatBoost final com alpha={alpha:.2f}")
        previsoes_brutas, duracao = treinar_e_prever_teste(
            df_treino,
            df_teste,
            alpha,
        )
        previsoes_finais = aplicar_correspondencias(
            previsoes_brutas,
            previsoes_correspondencias,
            PESO_CORRESPONDENCIA,
        )

        sufixo = nome_alpha(alpha)
        previsoes_teste[f"catboost_{sufixo}_bruto"] = previsoes_brutas
        previsoes_teste[f"catboost_{sufixo}_corr30"] = previsoes_finais

        submission = modelo_resposta.copy()
        submission["preco"] = previsoes_finais.round(2)
        arquivo = f"submission_catboost_{sufixo}_corr30.csv"
        caminho = os.path.join(pasta_submissions, arquivo)
        submission.to_csv(caminho, index=False)

        assert submission.shape == (2000, 2)
        assert submission["Id"].is_unique
        assert not submission.isna().any().any()
        assert (submission["preco"] > 0).all()
        print(f"  Tempo: {duracao:.1f}s")
        print(f"  Submission: {caminho}")

    caminho_previsoes = os.path.join(
        pasta_resultados,
        "catboost_previsoes_teste_alphas_corr30.csv",
    )
    previsoes_teste.to_csv(caminho_previsoes, index=False)
    print(f"Previsoes preservadas: {caminho_previsoes}")


if __name__ == "__main__":
    main()
