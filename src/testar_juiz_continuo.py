"""Testa um juiz continuo alinhado diretamente ao RMSPE do blend.

Para G=generalista, S=especialista, D=S-G e preco y, a perda por linha e:

    ((G + w*D - y) / y) ** 2
      = (D/y)**2 * (w - (y-G)/D)**2

Assim, uma regressao do pseudo-alvo (y-G)/D com peso (D/y)**2 equivale
ao objetivo quadratico do blend. Uma segunda variante projeta o pseudo-alvo
em [0, 1] antes do treino para testar maior robustez.
"""

import json
import os
import time

import numpy as np
import pandas as pd

from comparar_juizes_especialista import (
    criar_regressor,
    preparar_features,
    preparar_features_com_previsoes,
)
from gerar_submissions_juiz_especialista import (
    LIMITE_BARATO,
    PESO_CORRESPONDENCIA,
    carregar_bases_oof,
    rmspe,
)
from testar_catboost_alphas import COLUNAS_CATEGORICAS, carregar_treino
from testar_catboost_correspondencias import aplicar_correspondencias


VARIANTES = ["alvo_exato", "alvo_projetado_01"]


def construir_alvos(meta):
    y = meta["preco"].to_numpy(dtype=float)
    geral = meta["generalista"].to_numpy(dtype=float)
    especialista = meta["especialista"].to_numpy(dtype=float)
    diferenca = especialista - geral
    alvo_exato = np.divide(
        y - geral,
        diferenca,
        out=np.zeros_like(y),
        where=np.abs(diferenca) > 1e-12,
    )
    peso = np.square(diferenca / y)
    peso = np.maximum(peso, 1e-12)
    peso /= peso.mean()
    assert np.isfinite(alvo_exato).all()
    assert np.isfinite(peso).all()
    return {
        "alvo_exato": alvo_exato,
        "alvo_projetado_01": np.clip(alvo_exato, 0.0, 1.0),
    }, peso


def treinar_scores_crossfit(x, meta, alvos, peso):
    scores = {
        nome: np.full(len(meta), np.nan, dtype=float) for nome in VARIANTES
    }
    tempos = {nome: 0.0 for nome in VARIANTES}
    for fold in sorted(meta["fold"].unique()):
        treino = meta["fold"].ne(fold).to_numpy()
        validacao = meta["fold"].eq(fold).to_numpy()
        for nome in VARIANTES:
            modelo = criar_regressor()
            inicio = time.perf_counter()
            modelo.fit(
                x.loc[treino],
                alvos[nome][treino],
                cat_features=COLUNAS_CATEGORICAS,
                sample_weight=peso[treino],
            )
            tempos[nome] += time.perf_counter() - inicio
            scores[nome][validacao] = modelo.predict(x.loc[validacao])
        print(f"Juizes continuos do fold {fold}/5 concluidos")
    assert all(np.isfinite(score).all() for score in scores.values())
    return scores, tempos


def aplicar_mapeamento(score, intercepto, inclinacao):
    return np.clip(intercepto + inclinacao * score, 0.0, 1.0)


def escolher_mapeamento(meta, score, mascara):
    y = meta.loc[mascara, "preco"].to_numpy()
    geral = meta.loc[mascara, "generalista"].to_numpy()
    especialista = meta.loc[mascara, "especialista"].to_numpy()
    score_treino = score[mascara]
    melhor = None
    for intercepto in np.linspace(-0.50, 0.50, 41):
        for inclinacao in np.linspace(0.0, 3.0, 61):
            peso = aplicar_mapeamento(score_treino, intercepto, inclinacao)
            pred = geral + peso * (especialista - geral)
            erro = rmspe(y, pred)
            if melhor is None or erro < melhor["rmspe"]:
                melhor = {
                    "intercepto": float(intercepto),
                    "inclinacao": float(inclinacao),
                    "rmspe": float(erro),
                }
    return melhor


def avaliar_crossfit(meta, scores, tempos):
    y = meta["preco"].to_numpy()
    geral = meta["generalista"].to_numpy()
    especialista = meta["especialista"].to_numpy()
    saidas = {}
    resumo = []
    parametros_folds = []
    for nome, score in scores.items():
        peso_cf = np.full(len(meta), np.nan)
        for fold in sorted(meta["fold"].unique()):
            treino = meta["fold"].ne(fold).to_numpy()
            validacao = meta["fold"].eq(fold).to_numpy()
            melhor = escolher_mapeamento(meta, score, treino)
            peso_cf[validacao] = aplicar_mapeamento(
                score[validacao],
                melhor["intercepto"],
                melhor["inclinacao"],
            )
            parametros_folds.append({"variante": nome, "fold": int(fold), **melhor})
        peso_direto = np.clip(score, 0.0, 1.0)
        pred_direto = geral + peso_direto * (especialista - geral)
        pred_cf = geral + peso_cf * (especialista - geral)
        melhor_final = escolher_mapeamento(
            meta, score, np.ones(len(meta), dtype=bool)
        )
        fold_scores = {}
        for fold in sorted(meta["fold"].unique()):
            mascara = meta["fold"].eq(fold).to_numpy()
            fold_scores[f"rmspe_fold_{int(fold)}"] = rmspe(
                y[mascara], pred_cf[mascara]
            )
        resumo.append(
            {
                "variante": nome,
                "rmspe_peso_direto": rmspe(y, pred_direto),
                "rmspe_calibrado_crossfit": rmspe(y, pred_cf),
                "peso_medio_crossfit": peso_cf.mean(),
                "peso_mediano_crossfit": np.median(peso_cf),
                "fracao_peso_zero": np.mean(peso_cf == 0),
                "fracao_peso_um": np.mean(peso_cf == 1),
                "intercepto_final": melhor_final["intercepto"],
                "inclinacao_final": melhor_final["inclinacao"],
                "tempo_cv_segundos": tempos[nome],
                **fold_scores,
            }
        )
        saidas[nome] = {
            "score": score,
            "peso_direto": peso_direto,
            "peso_cf": peso_cf,
            "pred_cf": pred_cf,
            "parametro_final": melhor_final,
        }
    return saidas, pd.DataFrame(resumo), pd.DataFrame(parametros_folds)


def preparar_holdout(raiz, df):
    detalhe = pd.read_csv(
        os.path.join(raiz, "resultados", "juiz_especialista_holdout_id.csv")
    )
    cat = pd.read_csv(
        os.path.join(
            raiz, "resultados", "catboost_correspondencias_holdout_id.csv"
        )
    )
    assert np.array_equal(detalhe["Id"].to_numpy(), cat["Id"].to_numpy())
    ids = set(detalhe["Id"])
    mascara_validacao = df["Id"].isin(ids).to_numpy()
    validacao = df.loc[mascara_validacao]
    assert np.array_equal(validacao["Id"].to_numpy(), detalhe["Id"].to_numpy())
    pred_cat = aplicar_correspondencias(
        cat["catboost_alpha_1.50"].to_numpy(),
        cat["previsao_correspondencia"].to_numpy(),
        PESO_CORRESPONDENCIA,
    )
    pred_geral = detalhe["generalista"].to_numpy()
    pred_especialista = detalhe["especialista"].to_numpy()
    pred_arvores = (pred_geral - 0.60 * pred_cat) / 0.40
    x_validacao = preparar_features_com_previsoes(
        validacao,
        pred_geral,
        pred_especialista,
        pred_cat,
        pred_arvores,
        cat["previsao_correspondencia"].notna(),
    )
    return detalhe, x_validacao, mascara_validacao


def auditar_holdout(
    raiz,
    df,
    x_oof,
    meta,
    alvos,
    peso_treino,
    scores_oof,
):
    detalhe, x_validacao, mascara_validacao = preparar_holdout(raiz, df)
    mascara_treino = ~mascara_validacao
    y = detalhe["preco"].to_numpy()
    geral = detalhe["generalista"].to_numpy()
    especialista = detalhe["especialista"].to_numpy()
    resultados = []
    detalhes_saida = detalhe.copy()
    for nome in VARIANTES:
        modelo = criar_regressor()
        inicio = time.perf_counter()
        modelo.fit(
            x_oof.loc[mascara_treino],
            alvos[nome][mascara_treino],
            cat_features=COLUNAS_CATEGORICAS,
            sample_weight=peso_treino[mascara_treino],
        )
        tempo = time.perf_counter() - inicio
        score = modelo.predict(x_validacao)
        parametros = escolher_mapeamento(meta, scores_oof[nome], mascara_treino)
        peso = aplicar_mapeamento(
            score, parametros["intercepto"], parametros["inclinacao"]
        )
        pred = geral + peso * (especialista - geral)
        baratos = y <= LIMITE_BARATO
        resultados.append(
            {
                "variante": nome,
                "rmspe_holdout": rmspe(y, pred),
                "rmspe_baratos_ate_355k": rmspe(y[baratos], pred[baratos]),
                "rmspe_caros_acima_355k": rmspe(y[~baratos], pred[~baratos]),
                "peso_medio": peso.mean(),
                "peso_mediano": np.median(peso),
                "fracao_peso_zero": np.mean(peso == 0),
                "fracao_peso_um": np.mean(peso == 1),
                "intercepto_treino": parametros["intercepto"],
                "inclinacao_treino": parametros["inclinacao"],
                "tempo_treino_segundos": tempo,
            }
        )
        detalhes_saida[f"score_{nome}"] = score
        detalhes_saida[f"peso_{nome}"] = peso
        detalhes_saida[f"pred_{nome}"] = pred
    resultados = pd.DataFrame(resultados).sort_values("rmspe_holdout")
    pasta = os.path.join(raiz, "resultados")
    resultados.to_csv(
        os.path.join(pasta, "juiz_continuo_holdout_id_resumo.csv"), index=False
    )
    detalhes_saida.to_csv(
        os.path.join(pasta, "juiz_continuo_holdout_id.csv"), index=False
    )
    return resultados


def main():
    raiz, df = carregar_treino()
    geral, especialista, cat, meta, x = carregar_bases_oof(raiz, df)
    alvos, peso = construir_alvos(meta)
    scores, tempos = treinar_scores_crossfit(x, meta, alvos, peso)
    saidas, resumo, parametros_folds = avaliar_crossfit(meta, scores, tempos)
    resumo = resumo.sort_values("rmspe_calibrado_crossfit")

    pasta = os.path.join(raiz, "resultados")
    oof = meta[["Id", "preco", "fold", "generalista", "especialista"]].copy()
    for nome, dados in saidas.items():
        oof[f"score_{nome}"] = dados["score"]
        oof[f"peso_direto_{nome}"] = dados["peso_direto"]
        oof[f"peso_cf_{nome}"] = dados["peso_cf"]
        oof[f"pred_cf_{nome}"] = dados["pred_cf"]
    oof.to_csv(os.path.join(pasta, "juiz_continuo_oof.csv"), index=False)
    resumo.to_csv(os.path.join(pasta, "juiz_continuo_resumo.csv"), index=False)
    parametros_folds.to_csv(
        os.path.join(pasta, "juiz_continuo_parametros_folds.csv"), index=False
    )
    parametros_finais = {
        nome: dados["parametro_final"] for nome, dados in saidas.items()
    }
    with open(
        os.path.join(pasta, "juiz_continuo_parametros_finais.json"),
        "w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(parametros_finais, arquivo, ensure_ascii=False, indent=2)

    print("\nResumo cross-fit:")
    print(resumo.to_string(index=False))
    print("\nAuditando no holdout por IDs...")
    holdout = auditar_holdout(
        raiz, df, x, meta, alvos, peso, scores
    )
    print(holdout.to_string(index=False))

    melhor_cf = resumo.iloc[0]
    melhor_holdout = holdout.set_index("variante").loc[melhor_cf["variante"]]
    elegivel = bool(
        melhor_cf["rmspe_calibrado_crossfit"] <= 0.2130
        and melhor_holdout["rmspe_holdout"] < 0.21003540999791256
    )
    decisao = {
        "variante_vencedora": melhor_cf["variante"],
        "rmspe_crossfit": float(melhor_cf["rmspe_calibrado_crossfit"]),
        "rmspe_holdout": float(melhor_holdout["rmspe_holdout"]),
        "criterio_crossfit_maximo": 0.2130,
        "referencia_holdout_juiz_binario": 0.21003540999791256,
        "elegivel_para_submission": elegivel,
    }
    with open(
        os.path.join(pasta, "juiz_continuo_decisao.json"),
        "w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(decisao, arquivo, ensure_ascii=False, indent=2)
    print("\nDecisao:")
    print(json.dumps(decisao, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
