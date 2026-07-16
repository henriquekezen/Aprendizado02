"""Testa curvas de iteracoes do CatBoost barato log alpha 2."""

import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from buscar_catboost_baratos import (
    LIMITES_PRECO,
    LIMITE_PRINCIPAL,
    calibrar_crossfit,
    calcular_fator_otimo,
    completar_configuracao,
    criar_modelo,
    preparar_alvo,
    restaurar_previsao,
)
from testar_catboost_alphas import (
    COLUNAS_CATEGORICAS,
    calcular_pesos,
    calcular_rmspe,
    carregar_treino,
    separar_xy,
)


FINS_ARVORES_PADRAO = [400, 600, 800, 1000, 1200, 1400, 1600]


def executar(depth, random_strength, fins_arvores, sufixo):
    raiz, df = carregar_treino()
    splits = list(KFold(n_splits=5, shuffle=True, random_state=42).split(df))
    df = df.copy()
    df["fold_busca"] = 0
    for numero_fold, (_, indices_validacao) in enumerate(splits, start=1):
        df.loc[indices_validacao, "fold_busca"] = numero_fold

    configuracao = completar_configuracao(
        {
            "nome": "log_a200_d6_itercurve",
            "target": "log",
            "alpha": 2.0,
            "depth": depth,
            "random_strength": random_strength,
            "iterations": max(fins_arvores),
        }
    )
    previsoes_oof = {
        fim: np.full(len(df), np.nan) for fim in fins_arvores
    }
    folds_detalhados = []
    duracao_total = 0.0

    for numero_fold, (indices_treino, indices_validacao) in enumerate(
        splits,
        start=1,
    ):
        treino = df.iloc[indices_treino].drop(columns=["fold_busca"])
        validacao = df.iloc[indices_validacao].drop(columns=["fold_busca"])
        x_treino, y_treino = separar_xy(treino)
        x_validacao, _ = separar_xy(validacao)
        modelo = criar_modelo(configuracao)
        inicio = time.perf_counter()
        modelo.fit(
            x_treino,
            preparar_alvo(y_treino, "log"),
            cat_features=COLUNAS_CATEGORICAS,
            sample_weight=calcular_pesos(y_treino, 2.0),
        )
        duracao = time.perf_counter() - inicio
        duracao_total += duracao

        for fim in fins_arvores:
            previsoes = restaurar_previsao(
                modelo.predict(x_validacao, ntree_end=fim),
                "log",
            )
            previsoes_oof[fim][indices_validacao] = previsoes
            mascara = validacao["preco"] <= LIMITE_PRINCIPAL
            folds_detalhados.append(
                {
                    "iteracoes": fim,
                    "fold": numero_fold,
                    "rmspe_ate_355k": calcular_rmspe(
                        validacao.loc[mascara, "preco"],
                        previsoes[mascara.to_numpy()],
                    ),
                }
            )
        print(
            f"Fold {numero_fold}/5 concluido - "
            f"treino de {max(fins_arvores)} arvores: {duracao:.1f}s"
        )

    resumos = []
    curvas = []
    oof = df[["Id", "preco", "fold_busca"]].copy()
    mascara_principal = df["preco"] <= LIMITE_PRINCIPAL
    for fim in fins_arvores:
        assert np.isfinite(previsoes_oof[fim]).all()
        calibradas, fatores = calibrar_crossfit(df, previsoes_oof[fim])
        fator_final = calcular_fator_otimo(
            df.loc[mascara_principal, "preco"],
            previsoes_oof[fim][mascara_principal.to_numpy()],
        )
        resumos.append(
            {
                "iteracoes": fim,
                "rmspe_ate_355k": calcular_rmspe(
                    df.loc[mascara_principal, "preco"],
                    previsoes_oof[fim][mascara_principal.to_numpy()],
                ),
                "rmspe_ate_355k_calibrado_cf": calcular_rmspe(
                    df.loc[mascara_principal, "preco"],
                    calibradas[mascara_principal.to_numpy()],
                ),
                "fator_calibracao_final": fator_final,
                "fator_min_fold": min(fatores.values()),
                "fator_max_fold": max(fatores.values()),
                "duracao_treino_max_cv_segundos": duracao_total,
            }
        )
        for limite in LIMITES_PRECO:
            mascara = df["preco"] <= limite
            curvas.append(
                {
                    "iteracoes": fim,
                    "limite_preco": limite,
                    "quantidade": int(mascara.sum()),
                    "rmspe": calcular_rmspe(
                        df.loc[mascara, "preco"],
                        previsoes_oof[fim][mascara.to_numpy()],
                    ),
                    "rmspe_calibrado_cf": calcular_rmspe(
                        df.loc[mascara, "preco"],
                        calibradas[mascara.to_numpy()],
                    ),
                }
            )
        oof[f"iter_{fim}_bruto"] = previsoes_oof[fim]
        oof[f"iter_{fim}_calibrado_cf"] = calibradas

    pasta = os.path.join(raiz, "resultados")
    prefixo = f"catboost_baratos_curva_iteracoes_{sufixo}"
    resumo_df = pd.DataFrame(resumos).sort_values(
        "rmspe_ate_355k_calibrado_cf"
    )
    resumo_df.to_csv(os.path.join(pasta, f"{prefixo}_resumo.csv"), index=False)
    pd.DataFrame(curvas).to_csv(
        os.path.join(pasta, f"{prefixo}_curvas.csv"), index=False
    )
    pd.DataFrame(folds_detalhados).to_csv(
        os.path.join(pasta, f"{prefixo}_folds.csv"), index=False
    )
    oof.to_csv(os.path.join(pasta, f"{prefixo}_oof.csv"), index=False)

    print("\nResumo da curva de iteracoes:")
    print(resumo_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--random-strength", type=float, default=1.0)
    parser.add_argument(
        "--fins-arvores",
        type=int,
        nargs="+",
        default=FINS_ARVORES_PADRAO,
    )
    parser.add_argument("--sufixo", default="d6_rs1")
    args = parser.parse_args()
    executar(
        args.depth,
        args.random_strength,
        args.fins_arvores,
        args.sufixo,
    )


if __name__ == "__main__":
    main()
