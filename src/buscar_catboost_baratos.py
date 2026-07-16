"""Busca iterativa de CatBoost especializado em imoveis baratos.

Cada rodada recebe configuracoes nomeadas, gera previsoes out-of-fold nos
mesmos cinco folds e mede o RMSPE cumulativo por limite de preco. Tambem
avalia uma calibracao multiplicativa treinada fora do fold avaliado.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

from testar_catboost_alphas import (
    COLUNAS_CATEGORICAS,
    calcular_pesos,
    calcular_rmspe,
    carregar_treino,
    separar_xy,
)


LIMITE_PRINCIPAL = 355_000
LIMITES_PRECO = [250_000, 300_000, 355_000, 400_000, 500_000, 830_000]

RODADAS = {
    "alphas_altos": [
        {
            "nome": "log_a200_d6",
            "target": "log",
            "alpha": 2.0,
        },
        {
            "nome": "log_a250_d6",
            "target": "log",
            "alpha": 2.5,
        },
        {
            "nome": "log_a300_d6",
            "target": "log",
            "alpha": 3.0,
        },
    ],
    "alvo_bruto": [
        {
            "nome": "raw_a200_d6",
            "target": "raw_100k",
            "alpha": 2.0,
        },
        {
            "nome": "raw_a250_d6",
            "target": "raw_100k",
            "alpha": 2.5,
        },
        {
            "nome": "raw_a300_d6",
            "target": "raw_100k",
            "alpha": 3.0,
        },
    ],
    "capacidade": [
        {
            "nome": "log_a200_d5_i1200",
            "target": "log",
            "alpha": 2.0,
            "depth": 5,
            "iterations": 1200,
        },
        {
            "nome": "log_a200_d7_i800",
            "target": "log",
            "alpha": 2.0,
            "depth": 7,
            "iterations": 800,
        },
        {
            "nome": "log_a200_d6_l2_10",
            "target": "log",
            "alpha": 2.0,
            "l2_leaf_reg": 10.0,
        },
    ],
    "aleatoriedade": [
        {
            "nome": "log_a200_d6_rs0",
            "target": "log",
            "alpha": 2.0,
            "random_strength": 0.0,
        },
        {
            "nome": "log_a200_d6_bt0",
            "target": "log",
            "alpha": 2.0,
            "bagging_temperature": 0.0,
        },
        {
            "nome": "log_a200_d6_rs0_bt0",
            "target": "log",
            "alpha": 2.0,
            "random_strength": 0.0,
            "bagging_temperature": 0.0,
        },
    ],
    "refino_random": [
        {
            "nome": "log_a200_d6_rs025",
            "target": "log",
            "alpha": 2.0,
            "random_strength": 0.25,
        },
        {
            "nome": "log_a200_d6_rs050",
            "target": "log",
            "alpha": 2.0,
            "random_strength": 0.50,
        },
        {
            "nome": "log_a200_d7_i800_rs0",
            "target": "log",
            "alpha": 2.0,
            "depth": 7,
            "iterations": 800,
            "random_strength": 0.0,
        },
    ],
}


def completar_configuracao(configuracao):
    padrao = {
        "iterations": 1000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 5.0,
        "random_strength": 1.0,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 1.0,
        "boosting_type": "Ordered",
    }
    return {**padrao, **configuracao}


def criar_modelo(configuracao, semente=42):
    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=configuracao["iterations"],
        learning_rate=configuracao["learning_rate"],
        depth=configuracao["depth"],
        l2_leaf_reg=configuracao["l2_leaf_reg"],
        random_strength=configuracao["random_strength"],
        bootstrap_type=configuracao["bootstrap_type"],
        bagging_temperature=configuracao["bagging_temperature"],
        boosting_type=configuracao["boosting_type"],
        random_seed=semente,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )


def preparar_alvo(y_preco, target):
    if target == "log":
        return np.log1p(y_preco)
    if target == "raw_100k":
        return y_preco / 100_000.0
    raise ValueError(f"Target desconhecido: {target}")


def restaurar_previsao(previsoes, target):
    if target == "log":
        return np.expm1(previsoes)
    if target == "raw_100k":
        return previsoes * 100_000.0
    raise ValueError(f"Target desconhecido: {target}")


def treinar_fold(df_treino, df_validacao, configuracao):
    colunas_auxiliares = ["fold_busca"]
    x_treino, y_treino = separar_xy(
        df_treino.drop(columns=colunas_auxiliares, errors="ignore")
    )
    x_validacao, _ = separar_xy(
        df_validacao.drop(columns=colunas_auxiliares, errors="ignore")
    )
    pesos = calcular_pesos(y_treino, configuracao["alpha"])
    modelo = criar_modelo(configuracao)
    inicio = time.perf_counter()
    modelo.fit(
        x_treino,
        preparar_alvo(y_treino, configuracao["target"]),
        cat_features=COLUNAS_CATEGORICAS,
        sample_weight=pesos,
    )
    duracao = time.perf_counter() - inicio
    previsoes = restaurar_previsao(
        modelo.predict(x_validacao),
        configuracao["target"],
    )
    return previsoes, duracao


def calcular_fator_otimo(y_real, previsoes):
    razao = np.asarray(previsoes) / np.asarray(y_real)
    return razao.sum() / np.square(razao).sum()


def calibrar_crossfit(df, previsoes):
    mascara_baratos = df["preco"] <= LIMITE_PRINCIPAL
    previsoes_calibradas = previsoes.copy()
    fatores = {}
    for fold in sorted(df["fold_busca"].unique()):
        mascara_ajuste = mascara_baratos & df["fold_busca"].ne(fold)
        fator = calcular_fator_otimo(
            df.loc[mascara_ajuste, "preco"],
            previsoes[mascara_ajuste.to_numpy()],
        )
        fatores[int(fold)] = fator
        previsoes_calibradas[df["fold_busca"].eq(fold).to_numpy()] *= fator
    return previsoes_calibradas, fatores


def avaliar_configuracao(df, configuracao, folds):
    previsoes_oof = np.full(len(df), np.nan)
    duracao_total = 0.0
    resultados_folds = []

    for numero_fold, (indices_treino, indices_validacao) in enumerate(
        folds,
        start=1,
    ):
        previsoes, duracao = treinar_fold(
            df.iloc[indices_treino],
            df.iloc[indices_validacao],
            configuracao,
        )
        previsoes_oof[indices_validacao] = previsoes
        duracao_total += duracao
        mascara_baratos = df.iloc[indices_validacao]["preco"] <= LIMITE_PRINCIPAL
        resultados_folds.append(
            {
                "nome": configuracao["nome"],
                "fold": numero_fold,
                "quantidade_baratos": int(mascara_baratos.sum()),
                "rmspe_ate_355k": calcular_rmspe(
                    df.iloc[indices_validacao].loc[mascara_baratos, "preco"],
                    previsoes[mascara_baratos.to_numpy()],
                ),
                "duracao_segundos": duracao,
            }
        )
        print(
            f"  {configuracao['nome']} - fold {numero_fold}/5 - "
            f"baratos={resultados_folds[-1]['rmspe_ate_355k'] * 100:.2f}%"
        )

    assert np.isfinite(previsoes_oof).all()
    previsoes_calibradas, fatores = calibrar_crossfit(df, previsoes_oof)
    mascara_principal = df["preco"] <= LIMITE_PRINCIPAL
    fator_final = calcular_fator_otimo(
        df.loc[mascara_principal, "preco"],
        previsoes_oof[mascara_principal.to_numpy()],
    )

    resumo = {
        **configuracao,
        "rmspe_ate_355k": calcular_rmspe(
            df.loc[mascara_principal, "preco"],
            previsoes_oof[mascara_principal.to_numpy()],
        ),
        "rmspe_ate_355k_calibrado_cf": calcular_rmspe(
            df.loc[mascara_principal, "preco"],
            previsoes_calibradas[mascara_principal.to_numpy()],
        ),
        "fator_calibracao_final": fator_final,
        "fator_calibracao_min_fold": min(fatores.values()),
        "fator_calibracao_max_fold": max(fatores.values()),
        "media_razao_previsto_real_ate_355k": np.mean(
            previsoes_oof[mascara_principal.to_numpy()]
            / df.loc[mascara_principal, "preco"].to_numpy()
        ),
        "duracao_cv_segundos": duracao_total,
    }
    curvas = []
    for limite in LIMITES_PRECO:
        mascara = df["preco"] <= limite
        curvas.append(
            {
                "nome": configuracao["nome"],
                "limite_preco": limite,
                "quantidade": int(mascara.sum()),
                "rmspe": calcular_rmspe(
                    df.loc[mascara, "preco"],
                    previsoes_oof[mascara.to_numpy()],
                ),
                "rmspe_calibrado_cf": calcular_rmspe(
                    df.loc[mascara, "preco"],
                    previsoes_calibradas[mascara.to_numpy()],
                ),
            }
        )
    return previsoes_oof, previsoes_calibradas, resumo, curvas, resultados_folds


def executar_rodada(nome_rodada):
    if nome_rodada not in RODADAS:
        raise ValueError(f"Rodada desconhecida: {nome_rodada}")
    raiz, df = carregar_treino()
    splits = list(KFold(n_splits=5, shuffle=True, random_state=42).split(df))
    df = df.copy()
    df["fold_busca"] = 0
    for numero_fold, (_, indices_validacao) in enumerate(splits, start=1):
        df.loc[indices_validacao, "fold_busca"] = numero_fold

    resumos = []
    curvas = []
    folds_detalhados = []
    oof = df[["Id", "preco", "fold_busca"]].copy()

    for configuracao_bruta in RODADAS[nome_rodada]:
        configuracao = completar_configuracao(configuracao_bruta)
        print(f"\nAvaliando {configuracao['nome']}")
        (
            previsoes,
            previsoes_calibradas,
            resumo,
            curvas_configuracao,
            folds_configuracao,
        ) = avaliar_configuracao(df, configuracao, splits)
        oof[f"{configuracao['nome']}_bruto"] = previsoes
        oof[f"{configuracao['nome']}_calibrado_cf"] = previsoes_calibradas
        resumos.append(resumo)
        curvas.extend(curvas_configuracao)
        folds_detalhados.extend(folds_configuracao)

    pasta = os.path.join(raiz, "resultados")
    os.makedirs(pasta, exist_ok=True)
    prefixo = f"catboost_baratos_{nome_rodada}"
    pd.DataFrame(resumos).sort_values("rmspe_ate_355k_calibrado_cf").to_csv(
        os.path.join(pasta, f"{prefixo}_resumo.csv"),
        index=False,
    )
    pd.DataFrame(curvas).to_csv(
        os.path.join(pasta, f"{prefixo}_curvas.csv"),
        index=False,
    )
    pd.DataFrame(folds_detalhados).to_csv(
        os.path.join(pasta, f"{prefixo}_folds.csv"),
        index=False,
    )
    oof.to_csv(
        os.path.join(pasta, f"{prefixo}_oof.csv"),
        index=False,
    )
    with open(
        os.path.join(pasta, f"{prefixo}_configuracoes.json"),
        "w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(
            [completar_configuracao(c) for c in RODADAS[nome_rodada]],
            arquivo,
            ensure_ascii=False,
            indent=2,
        )

    resumo_df = pd.DataFrame(resumos).sort_values(
        "rmspe_ate_355k_calibrado_cf"
    )
    print("\nResumo da rodada:")
    print(
        resumo_df[
            [
                "nome",
                "rmspe_ate_355k",
                "rmspe_ate_355k_calibrado_cf",
                "fator_calibracao_final",
                "media_razao_previsto_real_ate_355k",
            ]
        ].to_string(index=False)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rodada", choices=sorted(RODADAS))
    args = parser.parse_args()
    executar_rodada(args.rodada)


if __name__ == "__main__":
    main()
