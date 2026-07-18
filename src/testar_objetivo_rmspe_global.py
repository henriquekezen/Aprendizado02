"""Testa o objetivo RMSPE exato em modelos globais.

A busca do especialista caro encontrou que treinar o preco bruto dividido por
1 milhao com `sample_weight = 1/preco^2` alinha a perda quadratica ponderada
ao erro percentual quadratico do RMSPE. Esse desenho nunca foi testado
globalmente: o generalista usa alvo em log com alpha 1,5.

Este script treina XGBoost, LightGBM e CatBoost globais com o objetivo exato,
nos mesmos cinco folds do pipeline e com correspondencia de 30%, e avalia:

1. cada modelo isolado contra o OOF do pipeline publico de 21,19%;
2. blends aditivos `w * novo + (1-w) * pipeline`, com peso escolhido de forma
   aninhada (leave-one-fold-out);
3. o mesmo protocolo no holdout dos 20% menores IDs, com o peso congelado
   escolhido apenas no OOF.

Nenhuma submission e gerada.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from buscar_especialista_caros import criar_lgb, criar_xgb, preparar_arvores, rmspe
from testar_catboost_alphas import (
    COLUNAS_CATEGORICAS,
    carregar_treino,
    criar_modelo,
    separar_xy,
)
from testar_catboost_correspondencias import (
    aplicar_correspondencias,
    preparar_divisoes,
    preparar_holdout_id,
)


PESO_CORRESPONDENCIA = 0.30
ITERACOES_CATBOOST = 800
GRADE_PESOS_BLEND = np.round(np.arange(0.0, 0.65, 0.05), 2)
FAIXAS_PRECO = [
    ("ate_740k", 0.0, 740_000.0),
    ("740k_1m", 740_000.0, 1_000_000.0),
    ("1m_1.3m", 1_000_000.0, 1_300_000.0),
    ("1.3m_2m", 1_300_000.0, 2_000_000.0),
    ("acima_2m", 2_000_000.0, np.inf),
]


def pesos_rmspe(y):
    pesos = np.power(np.asarray(y, dtype=float), -2.0)
    return pesos / pesos.mean()


def alvo_bruto(y):
    return np.asarray(y, dtype=float) / 1_000_000.0


def restaurar_bruto(p):
    return np.maximum(np.asarray(p, dtype=float) * 1_000_000.0, 1.0)


def treinar_arvores_raw(df_treino, df_validacao):
    treino, validacao = preparar_arvores(df_treino, df_validacao)
    x_treino = treino.drop(columns=["Id", "preco"])
    x_validacao = validacao.drop(columns=["Id", "preco"], errors="ignore")
    y = treino["preco"].to_numpy(dtype=float)
    alvo = alvo_bruto(y)
    pesos = pesos_rmspe(y)

    xgb = criar_xgb()
    xgb.fit(x_treino, alvo, sample_weight=pesos)
    lgb = criar_lgb()
    lgb.fit(x_treino, alvo, sample_weight=pesos, categorical_feature=["bairro"])
    return (
        restaurar_bruto(xgb.predict(x_validacao)),
        restaurar_bruto(lgb.predict(x_validacao)),
    )


def treinar_catboost_raw(df_treino, df_validacao):
    x_treino, y_treino = separar_xy(df_treino)
    x_validacao, _ = separar_xy(df_validacao)
    y = y_treino.to_numpy(dtype=float)
    modelo = criar_modelo(ITERACOES_CATBOOST)
    modelo.fit(
        x_treino,
        alvo_bruto(y),
        cat_features=COLUNAS_CATEGORICAS,
        sample_weight=pesos_rmspe(y),
    )
    return restaurar_bruto(modelo.predict(x_validacao))


def carregar_base_pipeline(raiz, ids, arquivo):
    caminho = os.path.join(raiz, "resultados", arquivo)
    base = pd.read_csv(caminho)[["Id", "preco", "base"]]
    tabela = pd.DataFrame({"Id": ids}).merge(base, on="Id", how="left")
    assert tabela["base"].notna().all(), f"IDs sem base do pipeline em {arquivo}"
    return tabela["preco"].to_numpy(dtype=float), tabela["base"].to_numpy(dtype=float)


def avaliar_faixas(y, previsoes, nomes):
    linhas = []
    for rotulo, inferior, superior in FAIXAS_PRECO:
        mascara = (y >= inferior) & (y < superior)
        linha = {"faixa": rotulo, "n": int(mascara.sum())}
        for nome in nomes:
            linha[f"rmspe_{nome}"] = rmspe(y[mascara], previsoes[nome][mascara])
            linha[f"razao_media_{nome}"] = float(
                np.mean(previsoes[nome][mascara] / y[mascara])
            )
        linhas.append(linha)
    return pd.DataFrame(linhas)


def escolher_peso_blend(y, base, novo, mascara):
    melhor_peso, melhor_rmspe = 0.0, np.inf
    for peso in GRADE_PESOS_BLEND:
        atual = rmspe(y[mascara], (1 - peso) * base[mascara] + peso * novo[mascara])
        if atual < melhor_rmspe:
            melhor_peso, melhor_rmspe = float(peso), atual
    return melhor_peso, melhor_rmspe


def avaliar_blend_aninhado(y, base, novo, folds):
    previsao = base.copy()
    pesos_por_fold = {}
    for fold in sorted(np.unique(folds)):
        fora = folds != fold
        peso, _ = escolher_peso_blend(y, base, novo, fora)
        pesos_por_fold[int(fold)] = peso
        dentro = folds == fold
        previsao[dentro] = (1 - peso) * base[dentro] + peso * novo[dentro]
    return rmspe(y, previsao), pesos_por_fold, previsao


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sem-catboost",
        action="store_true",
        help="pula o CatBoost bruto (boosting Ordered e mais lento)",
    )
    argumentos = parser.parse_args()

    raiz, df = carregar_treino()
    divisoes, numero_fold, correspondencias_oof = preparar_divisoes(df)
    y_oof = df["preco"].to_numpy(dtype=float)
    _, base_oof = carregar_base_pipeline(
        raiz, df["Id"].to_numpy(), "juiz_caros_oof.csv"
    )

    nomes_modelos = ["xgb_raw", "lgb_raw"]
    if not argumentos.sem_catboost:
        nomes_modelos.append("cat_raw")
    brutos = {nome: np.full(len(df), np.nan) for nome in nomes_modelos}

    for fold, indices_treino, indices_validacao, correspondencias in divisoes:
        df_treino = df.iloc[indices_treino]
        df_validacao = df.iloc[indices_validacao]
        inicio = time.perf_counter()
        pred_xgb, pred_lgb = treinar_arvores_raw(df_treino, df_validacao)
        brutos["xgb_raw"][indices_validacao] = pred_xgb
        brutos["lgb_raw"][indices_validacao] = pred_lgb
        if "cat_raw" in brutos:
            brutos["cat_raw"][indices_validacao] = treinar_catboost_raw(
                df_treino, df_validacao
            )
        print(
            f"Fold {fold}/5 treinado em "
            f"{time.perf_counter() - inicio:.0f}s",
            flush=True,
        )

    previsoes = {"base": base_oof}
    for nome in nomes_modelos:
        assert np.isfinite(brutos[nome]).all()
        previsoes[nome] = aplicar_correspondencias(
            brutos[nome], correspondencias_oof, PESO_CORRESPONDENCIA
        )
    previsoes["media_arvores_raw"] = 0.5 * previsoes["xgb_raw"] + 0.5 * previsoes["lgb_raw"]
    if "cat_raw" in previsoes:
        previsoes["media_tres_raw"] = (
            previsoes["xgb_raw"] + previsoes["lgb_raw"] + previsoes["cat_raw"]
        ) / 3.0

    candidatos = [nome for nome in previsoes if nome != "base"]

    print("\n=== Modelos isolados no OOF (com correspondencia 30%) ===")
    resumo_isolados = []
    for nome in ["base"] + candidatos:
        erro_pct = (previsoes[nome] - y_oof) / y_oof
        correlacao = (
            float(np.corrcoef((base_oof - y_oof) / y_oof, erro_pct)[0, 1])
            if nome != "base"
            else 1.0
        )
        resumo_isolados.append(
            {
                "modelo": nome,
                "rmspe_oof": rmspe(y_oof, previsoes[nome]),
                "correlacao_erro_com_base": correlacao,
            }
        )
        print(
            f"{nome:>18}: {rmspe(y_oof, previsoes[nome]) * 100:.4f}% "
            f"(corr. erro com base: {correlacao:.3f})"
        )

    print("\n=== Blend aditivo sobre o pipeline (peso aninhado por fold) ===")
    resumo_blends = []
    melhor = None
    for nome in candidatos:
        rmspe_aninhado, pesos_por_fold, _ = avaliar_blend_aninhado(
            y_oof, base_oof, previsoes[nome], numero_fold
        )
        peso_global, rmspe_global = escolher_peso_blend(
            y_oof, base_oof, previsoes[nome], np.ones(len(df), dtype=bool)
        )
        ganho = (rmspe(y_oof, base_oof) - rmspe_aninhado) * 100
        resumo_blends.append(
            {
                "modelo": nome,
                "rmspe_blend_aninhado": rmspe_aninhado,
                "ganho_pp": ganho,
                "pesos_por_fold": pesos_por_fold,
                "peso_otimo_global_oof": peso_global,
                "rmspe_peso_global_oof": rmspe_global,
            }
        )
        print(
            f"{nome:>18}: aninhado {rmspe_aninhado * 100:.4f}% "
            f"(ganho {ganho:+.3f} p.p.), pesos por fold {pesos_por_fold}, "
            f"peso global {peso_global:.2f}"
        )
        if melhor is None or rmspe_aninhado < melhor[1]:
            melhor = (nome, rmspe_aninhado, peso_global)

    nome_melhor, _, peso_congelado = melhor
    print(
        f"\nMelhor candidato: {nome_melhor} "
        f"(peso congelado para o holdout: {peso_congelado:.2f})"
    )

    faixas = avaliar_faixas(y_oof, previsoes, ["base", nome_melhor])
    print("\n=== Comportamento por faixa de preco real (OOF) ===")
    print(faixas.to_string(index=False))

    print("\n=== Holdout pelos 20% menores IDs ===")
    df_treino_h, df_validacao_h, correspondencias_h = preparar_holdout_id(df)
    y_holdout, base_holdout = carregar_base_pipeline(
        raiz, df_validacao_h["Id"].to_numpy(), "juiz_caros_holdout.csv"
    )
    pred_xgb_h, pred_lgb_h = treinar_arvores_raw(df_treino_h, df_validacao_h)
    previsoes_holdout = {
        "base": base_holdout,
        "xgb_raw": aplicar_correspondencias(
            pred_xgb_h, correspondencias_h, PESO_CORRESPONDENCIA
        ),
        "lgb_raw": aplicar_correspondencias(
            pred_lgb_h, correspondencias_h, PESO_CORRESPONDENCIA
        ),
    }
    if "cat_raw" in brutos:
        previsoes_holdout["cat_raw"] = aplicar_correspondencias(
            treinar_catboost_raw(df_treino_h, df_validacao_h),
            correspondencias_h,
            PESO_CORRESPONDENCIA,
        )
        previsoes_holdout["media_tres_raw"] = (
            previsoes_holdout["xgb_raw"]
            + previsoes_holdout["lgb_raw"]
            + previsoes_holdout["cat_raw"]
        ) / 3.0
    previsoes_holdout["media_arvores_raw"] = (
        0.5 * previsoes_holdout["xgb_raw"] + 0.5 * previsoes_holdout["lgb_raw"]
    )

    resumo_holdout = {"rmspe_base": rmspe(y_holdout, base_holdout)}
    print(f"{'base':>18}: {resumo_holdout['rmspe_base'] * 100:.4f}%")
    for nome in candidatos:
        resumo_holdout[f"rmspe_{nome}"] = rmspe(y_holdout, previsoes_holdout[nome])
        print(f"{nome:>18}: {resumo_holdout[f'rmspe_{nome}'] * 100:.4f}%")
    blend_holdout = (
        1 - peso_congelado
    ) * base_holdout + peso_congelado * previsoes_holdout[nome_melhor]
    resumo_holdout["rmspe_blend_congelado"] = rmspe(y_holdout, blend_holdout)
    resumo_holdout["ganho_blend_pp"] = (
        resumo_holdout["rmspe_base"] - resumo_holdout["rmspe_blend_congelado"]
    ) * 100
    print(
        f"blend {nome_melhor} peso {peso_congelado:.2f}: "
        f"{resumo_holdout['rmspe_blend_congelado'] * 100:.4f}% "
        f"(ganho {resumo_holdout['ganho_blend_pp']:+.3f} p.p.)"
    )

    pasta = os.path.join(raiz, "resultados")
    oof = pd.DataFrame({"Id": df["Id"], "preco": y_oof, "fold": numero_fold})
    for nome, valores in previsoes.items():
        oof[nome] = valores
    oof.to_csv(
        os.path.join(pasta, "objetivo_rmspe_global_oof.csv"), index=False
    )
    holdout = pd.DataFrame({"Id": df_validacao_h["Id"], "preco": y_holdout})
    for nome, valores in previsoes_holdout.items():
        holdout[nome] = valores
    holdout.to_csv(
        os.path.join(pasta, "objetivo_rmspe_global_holdout.csv"), index=False
    )
    faixas.to_csv(
        os.path.join(pasta, "objetivo_rmspe_global_faixas.csv"), index=False
    )
    with open(
        os.path.join(pasta, "objetivo_rmspe_global_resumo.json"),
        "w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(
            {
                "protocolo": {
                    "alvo": "preco / 1e6",
                    "sample_weight": "1/preco^2 normalizado",
                    "correspondencia": PESO_CORRESPONDENCIA,
                    "grade_pesos_blend": GRADE_PESOS_BLEND.tolist(),
                    "referencia": "coluna base de juiz_caros_oof.csv (21,19% publico)",
                },
                "isolados_oof": resumo_isolados,
                "blends_oof": resumo_blends,
                "melhor_candidato": nome_melhor,
                "peso_congelado": peso_congelado,
                "holdout": resumo_holdout,
            },
            arquivo,
            ensure_ascii=False,
            indent=2,
        )
    print("\nArtefatos salvos em resultados/objetivo_rmspe_global_*.")


if __name__ == "__main__":
    main()
