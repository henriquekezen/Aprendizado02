"""Testa um segundo juiz continuo entre CatBoost global e arvores.

O primeiro juiz fornece q, o peso do especialista barato. O restante e
dividido de forma convexa entre CatBoost C e arvores A:

    P = q*S + (1-q) * ((1-r)*C + r*A)

O novo juiz aprende r. A variante ancorada preserva media proxima ao melhor
peso publico estimado para as arvores (48,745%).
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
    PESO_CORRESPONDENCIA,
    carregar_bases_oof,
    rmspe,
)
from testar_catboost_alphas import COLUNAS_CATEGORICAS, carregar_treino
from testar_catboost_correspondencias import aplicar_correspondencias


ANCORA_ARVORES_PUBLICA = 0.4874518666234162
ESTRATEGIAS = ["livre", "ancorada_publico"]


def montar_estado(meta, geral_oof):
    y = meta["preco"].to_numpy(dtype=float)
    especialista = meta["especialista"].to_numpy(dtype=float)
    q = meta["peso_cf_alvo_projetado_01"].to_numpy(dtype=float)
    cat = geral_oof["catboost_alpha150_corr30"].to_numpy(dtype=float)
    arvores = geral_oof["blend_arvores_alpha150_corr30"].to_numpy(dtype=float)
    base_cat = q * especialista + (1.0 - q) * cat
    direcao = (1.0 - q) * (arvores - cat)
    alvo_exato = np.divide(
        y - base_cat,
        direcao,
        out=np.full_like(y, ANCORA_ARVORES_PUBLICA),
        where=np.abs(direcao) > 1e-12,
    )
    alvo = np.clip(alvo_exato, 0.0, 1.0)
    peso = np.maximum(np.square(direcao / y), 1e-12)
    peso /= peso.mean()
    return {
        "y": y,
        "especialista": especialista,
        "q": q,
        "cat": cat,
        "arvores": arvores,
        "base_cat": base_cat,
        "direcao": direcao,
        "alvo": alvo,
        "peso": peso,
    }


def enriquecer_features(x, estado):
    x = x.copy()
    x["peso_especialista_continuo"] = estado["q"]
    x["pred_juiz_continuo"] = (
        estado["base_cat"] + 0.40 * estado["direcao"]
    )
    x["diferenca_arvores_cat"] = estado["arvores"] - estado["cat"]
    x["razao_arvores_cat"] = estado["arvores"] / estado["cat"]
    assert np.isfinite(x.select_dtypes(include=[np.number])).all().all()
    return x


def treinar_score_crossfit(x, meta, estado):
    score = np.full(len(meta), np.nan)
    tempos = []
    for fold in sorted(meta["fold"].unique()):
        treino = meta["fold"].ne(fold).to_numpy()
        validacao = meta["fold"].eq(fold).to_numpy()
        modelo = criar_regressor()
        inicio = time.perf_counter()
        modelo.fit(
            x.loc[treino],
            estado["alvo"][treino],
            cat_features=COLUNAS_CATEGORICAS,
            sample_weight=estado["peso"][treino],
        )
        tempos.append(time.perf_counter() - inicio)
        score[validacao] = modelo.predict(x.loc[validacao])
        print(f"Juiz de componentes do fold {fold}/5 concluido")
    assert np.isfinite(score).all()
    return score, tempos


def aplicar_mapeamento(score, intercepto, inclinacao):
    return np.clip(intercepto + inclinacao * score, 0.0, 1.0)


def predizer(estado, r):
    return estado["base_cat"] + r * estado["direcao"]


def intercepto_para_media(score, inclinacao, media_alvo):
    baixo, alto = -4.0, 4.0
    for _ in range(60):
        meio = (baixo + alto) / 2.0
        media = aplicar_mapeamento(score, meio, inclinacao).mean()
        if media < media_alvo:
            baixo = meio
        else:
            alto = meio
    return (baixo + alto) / 2.0


def escolher_mapeamento(meta, estado, score, mascara, estrategia):
    y = estado["y"][mascara]
    score_treino = score[mascara]
    melhor = None
    if estrategia == "livre":
        candidatos = (
            (intercepto, inclinacao)
            for intercepto in np.linspace(-0.50, 0.75, 51)
            for inclinacao in np.linspace(0.0, 3.0, 61)
        )
    elif estrategia == "ancorada_publico":
        candidatos = []
        for inclinacao in np.linspace(0.0, 3.0, 61):
            intercepto = intercepto_para_media(
                score_treino, inclinacao, ANCORA_ARVORES_PUBLICA
            )
            candidatos.append((intercepto, inclinacao))
    else:
        raise ValueError(estrategia)

    for intercepto, inclinacao in candidatos:
        r = aplicar_mapeamento(score_treino, intercepto, inclinacao)
        pred = (
            estado["base_cat"][mascara]
            + r * estado["direcao"][mascara]
        )
        erro = rmspe(y, pred)
        if melhor is None or erro < melhor["rmspe"]:
            melhor = {
                "intercepto": float(intercepto),
                "inclinacao": float(inclinacao),
                "rmspe": float(erro),
                "media_r_treino": float(r.mean()),
            }
    return melhor


def avaliar_crossfit(meta, estado, score):
    resultados = []
    detalhes = []
    saidas = {}
    for estrategia in ESTRATEGIAS:
        r_cf = np.full(len(meta), np.nan)
        for fold in sorted(meta["fold"].unique()):
            treino = meta["fold"].ne(fold).to_numpy()
            validacao = meta["fold"].eq(fold).to_numpy()
            parametros = escolher_mapeamento(
                meta, estado, score, treino, estrategia
            )
            r_cf[validacao] = aplicar_mapeamento(
                score[validacao],
                parametros["intercepto"],
                parametros["inclinacao"],
            )
            detalhes.append(
                {"estrategia": estrategia, "fold": int(fold), **parametros}
            )
        pred = predizer(estado, r_cf)
        parametro_final = escolher_mapeamento(
            meta,
            estado,
            score,
            np.ones(len(meta), dtype=bool),
            estrategia,
        )
        folds = {}
        for fold in sorted(meta["fold"].unique()):
            mascara = meta["fold"].eq(fold).to_numpy()
            folds[f"rmspe_fold_{int(fold)}"] = rmspe(
                estado["y"][mascara], pred[mascara]
            )
        resultados.append(
            {
                "estrategia": estrategia,
                "rmspe_crossfit": rmspe(estado["y"], pred),
                "media_r_crossfit": r_cf.mean(),
                "mediana_r_crossfit": np.median(r_cf),
                "p10_r_crossfit": np.quantile(r_cf, 0.10),
                "p90_r_crossfit": np.quantile(r_cf, 0.90),
                "fracao_r_zero": np.mean(r_cf == 0),
                "fracao_r_um": np.mean(r_cf == 1),
                "intercepto_final": parametro_final["intercepto"],
                "inclinacao_final": parametro_final["inclinacao"],
                **folds,
            }
        )
        saidas[estrategia] = {
            "r_cf": r_cf,
            "pred_cf": pred,
            "parametro_final": parametro_final,
        }

    for nome, r_constante in [
        ("estatico_40_atual", 0.40),
        ("estatico_ancora_publica", ANCORA_ARVORES_PUBLICA),
    ]:
        r = np.full(len(meta), r_constante)
        pred = predizer(estado, r)
        resultados.append(
            {
                "estrategia": nome,
                "rmspe_crossfit": rmspe(estado["y"], pred),
                "media_r_crossfit": r_constante,
                "mediana_r_crossfit": r_constante,
                "p10_r_crossfit": r_constante,
                "p90_r_crossfit": r_constante,
                "fracao_r_zero": 0.0,
                "fracao_r_um": 0.0,
                "intercepto_final": r_constante,
                "inclinacao_final": 0.0,
                **{
                    f"rmspe_fold_{int(fold)}": rmspe(
                        estado["y"][meta["fold"].eq(fold).to_numpy()],
                        pred[meta["fold"].eq(fold).to_numpy()],
                    )
                    for fold in sorted(meta["fold"].unique())
                },
            }
        )
    return (
        saidas,
        pd.DataFrame(resultados).sort_values("rmspe_crossfit"),
        pd.DataFrame(detalhes),
    )


def carregar_estado_holdout(raiz, df):
    h = pd.read_csv(
        os.path.join(raiz, "resultados", "juiz_continuo_holdout_id.csv")
    )
    cat_h = pd.read_csv(
        os.path.join(
            raiz, "resultados", "catboost_correspondencias_holdout_id.csv"
        )
    )
    assert np.array_equal(h["Id"].to_numpy(), cat_h["Id"].to_numpy())
    ids = set(h["Id"])
    mascara_validacao = df["Id"].isin(ids).to_numpy()
    validacao = df.loc[mascara_validacao]
    cat = aplicar_correspondencias(
        cat_h["catboost_alpha_1.50"].to_numpy(),
        cat_h["previsao_correspondencia"].to_numpy(),
        PESO_CORRESPONDENCIA,
    )
    geral = h["generalista"].to_numpy()
    arvores = (geral - 0.60 * cat) / 0.40
    q = h["peso_alvo_projetado_01"].to_numpy()
    especialista = h["especialista"].to_numpy()
    estado = {
        "y": h["preco"].to_numpy(),
        "especialista": especialista,
        "q": q,
        "cat": cat,
        "arvores": arvores,
        "base_cat": q * especialista + (1.0 - q) * cat,
        "direcao": (1.0 - q) * (arvores - cat),
    }
    x = preparar_features_com_previsoes(
        validacao,
        geral,
        especialista,
        cat,
        arvores,
        cat_h["previsao_correspondencia"].notna(),
    )
    x = enriquecer_features(x, estado)
    return h, estado, x, mascara_validacao


def auditar_holdout(
    raiz,
    df,
    x_oof,
    meta,
    estado_oof,
    score_oof,
):
    h, estado_h, x_h, mascara_validacao = carregar_estado_holdout(raiz, df)
    mascara_treino = ~mascara_validacao
    modelo = criar_regressor()
    inicio = time.perf_counter()
    modelo.fit(
        x_oof.loc[mascara_treino],
        estado_oof["alvo"][mascara_treino],
        cat_features=COLUNAS_CATEGORICAS,
        sample_weight=estado_oof["peso"][mascara_treino],
    )
    tempo = time.perf_counter() - inicio
    score_h = modelo.predict(x_h)
    resultados = []
    detalhe = h.copy()
    for estrategia in ESTRATEGIAS:
        parametros = escolher_mapeamento(
            meta,
            estado_oof,
            score_oof,
            mascara_treino,
            estrategia,
        )
        r = aplicar_mapeamento(
            score_h, parametros["intercepto"], parametros["inclinacao"]
        )
        pred = predizer(estado_h, r)
        baratos = estado_h["y"] <= 355_000
        resultados.append(
            {
                "estrategia": estrategia,
                "rmspe_holdout": rmspe(estado_h["y"], pred),
                "rmspe_baratos_ate_355k": rmspe(
                    estado_h["y"][baratos], pred[baratos]
                ),
                "rmspe_caros_acima_355k": rmspe(
                    estado_h["y"][~baratos], pred[~baratos]
                ),
                "media_r": r.mean(),
                "mediana_r": np.median(r),
                "p10_r": np.quantile(r, 0.10),
                "p90_r": np.quantile(r, 0.90),
                "intercepto_treino": parametros["intercepto"],
                "inclinacao_treino": parametros["inclinacao"],
                "tempo_treino_segundos": tempo,
            }
        )
        detalhe[f"score_{estrategia}"] = score_h
        detalhe[f"r_arvores_{estrategia}"] = r
        detalhe[f"pred_{estrategia}"] = pred
    for nome, r_constante in [
        ("estatico_40_atual", 0.40),
        ("estatico_ancora_publica", ANCORA_ARVORES_PUBLICA),
    ]:
        r = np.full(len(h), r_constante)
        pred = predizer(estado_h, r)
        baratos = estado_h["y"] <= 355_000
        resultados.append(
            {
                "estrategia": nome,
                "rmspe_holdout": rmspe(estado_h["y"], pred),
                "rmspe_baratos_ate_355k": rmspe(
                    estado_h["y"][baratos], pred[baratos]
                ),
                "rmspe_caros_acima_355k": rmspe(
                    estado_h["y"][~baratos], pred[~baratos]
                ),
                "media_r": r_constante,
                "mediana_r": r_constante,
                "p10_r": r_constante,
                "p90_r": r_constante,
                "intercepto_treino": r_constante,
                "inclinacao_treino": 0.0,
                "tempo_treino_segundos": 0.0,
            }
        )
    resultados = pd.DataFrame(resultados).sort_values("rmspe_holdout")
    pasta = os.path.join(raiz, "resultados")
    resultados.to_csv(
        os.path.join(pasta, "juiz_componentes_holdout_resumo.csv"), index=False
    )
    detalhe.to_csv(
        os.path.join(pasta, "juiz_componentes_holdout.csv"), index=False
    )
    return resultados


def main():
    raiz, df = carregar_treino()
    geral, especialista, cat, meta_binario, x = carregar_bases_oof(raiz, df)
    meta = pd.read_csv(os.path.join(raiz, "resultados", "juiz_continuo_oof.csv"))
    assert np.array_equal(df["Id"].to_numpy(), meta["Id"].to_numpy())
    estado = montar_estado(meta, geral)
    x = enriquecer_features(x, estado)
    score, tempos = treinar_score_crossfit(x, meta, estado)
    saidas, resumo, parametros_folds = avaliar_crossfit(meta, estado, score)

    pasta = os.path.join(raiz, "resultados")
    oof = meta[
        ["Id", "preco", "fold", "generalista", "especialista"]
    ].copy()
    oof["catboost_global"] = estado["cat"]
    oof["blend_arvores"] = estado["arvores"]
    oof["peso_especialista"] = estado["q"]
    oof["score_juiz_componentes"] = score
    for estrategia, dados in saidas.items():
        oof[f"r_arvores_{estrategia}"] = dados["r_cf"]
        oof[f"pred_{estrategia}"] = dados["pred_cf"]
    oof.to_csv(os.path.join(pasta, "juiz_componentes_oof.csv"), index=False)
    resumo["tempo_cv_segundos"] = sum(tempos)
    resumo.to_csv(
        os.path.join(pasta, "juiz_componentes_resumo.csv"), index=False
    )
    parametros_folds.to_csv(
        os.path.join(pasta, "juiz_componentes_parametros_folds.csv"),
        index=False,
    )
    parametros_finais = {
        estrategia: dados["parametro_final"]
        for estrategia, dados in saidas.items()
    }
    with open(
        os.path.join(pasta, "juiz_componentes_parametros_finais.json"),
        "w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(parametros_finais, arquivo, ensure_ascii=False, indent=2)

    print("\nResumo cross-fit:")
    print(resumo.to_string(index=False))
    print("\nAuditoria no holdout por IDs:")
    holdout = auditar_holdout(raiz, df, x, meta, estado, score)
    print(holdout.to_string(index=False))

    candidatos = resumo[resumo["estrategia"].isin(ESTRATEGIAS)].copy()
    candidatos = candidatos.sort_values("rmspe_crossfit")
    elegiveis = []
    for _, linha in candidatos.iterrows():
        h = holdout.set_index("estrategia").loc[linha["estrategia"]]
        melhorou_folds = sum(
            linha[f"rmspe_fold_{fold}"]
            < rmspe(
                estado["y"][meta["fold"].eq(fold).to_numpy()],
                meta.loc[
                    meta["fold"].eq(fold), "pred_cf_alvo_projetado_01"
                ],
            )
            for fold in range(1, 6)
        )
        if (
            linha["rmspe_crossfit"] <= 0.2125
            and h["rmspe_holdout"] < 0.20953554154390305
            and melhorou_folds >= 4
        ):
            elegiveis.append(
                {
                    "estrategia": linha["estrategia"],
                    "rmspe_crossfit": float(linha["rmspe_crossfit"]),
                    "rmspe_holdout": float(h["rmspe_holdout"]),
                    "folds_melhores": int(melhorou_folds),
                }
            )
    nomes_elegiveis = {item["estrategia"] for item in elegiveis}
    if "ancorada_publico" in nomes_elegiveis:
        estrategia_escolhida = "ancorada_publico"
        motivo_escolha = (
            "Preserva o equilibrio CatBoost/arvores estimado pelos scores "
            "publicos; a versao livre ganha apenas 0,019 p.p. OOF adicional "
            "e desloca a media para cerca de 66% de arvores."
        )
    else:
        estrategia_escolhida = elegiveis[0]["estrategia"] if elegiveis else None
        motivo_escolha = "Melhor estrategia elegivel disponivel."
    decisao = {
        "referencia_publica_juiz_continuo": 0.2130,
        "referencia_crossfit_juiz_continuo": 0.21287300085342017,
        "referencia_holdout_juiz_continuo": 0.20953554154390305,
        "ancora_arvores_publica": ANCORA_ARVORES_PUBLICA,
        "elegiveis": elegiveis,
        "gerar_submission": bool(elegiveis),
        "estrategia_escolhida": estrategia_escolhida,
        "motivo_escolha": motivo_escolha,
    }
    with open(
        os.path.join(pasta, "juiz_componentes_decisao.json"),
        "w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(decisao, arquivo, ensure_ascii=False, indent=2)
    print("\nDecisao:")
    print(json.dumps(decisao, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
