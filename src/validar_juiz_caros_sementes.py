"""Repete a validacao aninhada do juiz caro em novas sementes."""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from testar_catboost_alphas import carregar_treino
from testar_juiz_especialista_caros import (
    ESTRATEGIAS,
    avaliar_aninhado,
    componentes_oof,
    preparar_features,
)


SEMENTES_NOVAS = [7, 2026]


def criar_folds(quantidade, semente):
    folds = np.zeros(quantidade, dtype=int)
    cv = KFold(n_splits=5, shuffle=True, random_state=semente)
    for fold, (_, iv) in enumerate(cv.split(np.arange(quantidade)), start=1):
        folds[iv] = fold
    return folds


def executar():
    raiz, df = carregar_treino()
    pasta = os.path.join(raiz, "resultados")
    atual, exp, componentes = componentes_oof(pasta)
    x = preparar_features(
        df, componentes, exp["especialista_s42_calibrado_cf"]
    )
    meta_base = df[["Id", "preco"]].copy()
    meta_base["base"] = atual["pred_ancorada_publico"]
    meta_base["especialista_caro"] = exp["especialista_s42_calibrado_cf"]

    linhas = []
    resumo_42 = pd.read_csv(os.path.join(pasta, "juiz_caros_resumo.csv"))
    for _, linha in resumo_42.iterrows():
        linhas.append({
            "semente": 42,
            "estrategia": linha["estrategia"],
            "rmspe": linha["rmspe_crossfit_aninhado"],
            "ganho_pp_vs_base": linha["ganho_pp_vs_base"],
            "folds_melhores": sum(
                linha[f"ganho_fold_{fold}_pp"] > 0 for fold in range(1, 6)
            ),
        })

    for semente in SEMENTES_NOVAS:
        meta = meta_base.copy()
        meta["fold"] = criar_folds(len(meta), semente)
        _, _, resumo, _, _, duracao = avaliar_aninhado(x, meta)
        for _, linha in resumo.iterrows():
            linhas.append({
                "semente": semente,
                "estrategia": linha["estrategia"],
                "rmspe": linha["rmspe_crossfit_aninhado"],
                "ganho_pp_vs_base": linha["ganho_pp_vs_base"],
                "folds_melhores": sum(
                    linha[f"ganho_fold_{fold}_pp"] > 0
                    for fold in range(1, 6)
                ),
                "duracao_segundos": duracao,
            })
        print(f"semente {semente} concluida")

    detalhe = pd.DataFrame(linhas)
    detalhe.to_csv(
        os.path.join(pasta, "juiz_caros_estabilidade_sementes.csv"), index=False
    )
    agregado = (
        detalhe.groupby("estrategia", as_index=False)
        .agg(
            rmspe_medio=("rmspe", "mean"),
            desvio_rmspe=("rmspe", "std"),
            ganho_medio_pp=("ganho_pp_vs_base", "mean"),
            ganho_minimo_pp=("ganho_pp_vs_base", "min"),
            folds_melhores_total=("folds_melhores", "sum"),
        )
        .sort_values("rmspe_medio")
    )
    agregado.to_csv(
        os.path.join(pasta, "juiz_caros_estabilidade_resumo.csv"), index=False
    )
    print("\nEstabilidade em tres sementes:")
    print(agregado.to_string(index=False))


if __name__ == "__main__":
    executar()
