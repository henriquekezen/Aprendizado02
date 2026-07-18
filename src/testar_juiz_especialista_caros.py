"""Busca um juiz protegido para combinar o pipeline atual e o especialista caro.

O protocolo e aninhado. Em cada fold externo, scores internos OOF escolhem o
mapeamento de score para peso; depois os modelos sao treinados em todo o treino
externo e avaliados no fold nunca usado. O holdout por ID repete o processo.
"""

import json
import os
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import KFold

from gerar_submissions_juiz_especialista import carregar_teste, rmspe
from testar_catboost_alphas import (
    COLUNAS_CATEGORICAS,
    carregar_treino,
    criar_features_catboost,
)
from testar_catboost_correspondencias import aplicar_correspondencias


LIMITE_CARO = 1_000_000.0
PESO_CORRESPONDENCIA = 0.30
NOMES_SCORES = ["faixa_1m", "continuo_q95", "continuo_q99", "utilidade"]
ESTRATEGIAS = [
    "continuo_q99_livre",
    "continuo_q99_gate_faixa",
    "continuo_q99_gate_preco",
    "continuo_q99_gate_faixa_preco",
    "continuo_q95_gate_faixa",
    "continuo_q95_gate_preco",
    "utilidade_gate_faixa",
    "utilidade_gate_faixa_preco",
    "faixa_pura",
    "preco_puro",
]
FAIXAS_DIAGNOSTICO = [
    (0, 740_000),
    (740_000, 900_000),
    (900_000, 1_000_000),
    (1_000_000, 1_300_000),
    (1_300_000, 1_500_000),
    (1_500_000, 2_000_000),
    (2_000_000, np.inf),
]


def parametros_modelo():
    return {
        "n_estimators": 400,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 30,
        "subsample": 0.80,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_lambda": 2.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }


def criar_classificador():
    return LGBMClassifier(objective="binary", **parametros_modelo())


def criar_regressor():
    return LGBMRegressor(objective="regression", **parametros_modelo())


def alinhar_categorias(x_treino, x_previsao):
    treino = x_treino.copy()
    previsao = x_previsao.copy()
    for coluna in COLUNAS_CATEGORICAS:
        desconhecido = "__DESCONHECIDO__"
        categorias = pd.Index([
            *treino[coluna].astype(str).unique(), desconhecido
        ]).unique()
        valores_previsao = previsao[coluna].astype(str)
        valores_previsao = valores_previsao.where(
            valores_previsao.isin(categorias), desconhecido
        )
        treino[coluna] = pd.Categorical(
            treino[coluna].astype(str), categories=categorias
        )
        previsao[coluna] = pd.Categorical(
            valores_previsao, categories=categorias
        )
    return treino, previsao


def normalizar_pesos(valores, quantil=None):
    valores = np.asarray(valores, dtype=float).copy()
    if quantil is not None:
        valores = np.minimum(valores, np.quantile(valores, quantil))
    valores = np.maximum(valores, 1e-12)
    return valores / valores.mean()


def construir_alvos(y, base, especialista):
    y = np.asarray(y, dtype=float)
    base = np.asarray(base, dtype=float)
    especialista = np.asarray(especialista, dtype=float)
    direcao = especialista - base
    ideal = np.divide(
        y - base,
        direcao,
        out=np.zeros_like(y),
        where=np.abs(direcao) > 1e-9,
    )
    ideal = np.clip(ideal, 0.0, 1.0)
    perda_base = np.square((base - y) / y)
    perda_especialista = np.square((especialista - y) / y)
    delta = perda_base - perda_especialista
    return {
        "ideal": ideal,
        "peso_continuo": np.square(direcao / y),
        "utilidade": (delta > 0).astype(int),
        "peso_utilidade": np.abs(delta),
        "faixa_1m": (y >= LIMITE_CARO).astype(int),
    }


def treinar_score(nome, x_treino, alvos, x_previsao):
    x_treino, x_previsao = alinhar_categorias(x_treino, x_previsao)
    if nome == "faixa_1m":
        alvo = alvos["faixa_1m"]
        contagens = np.bincount(alvo, minlength=2).astype(float)
        pesos = len(alvo) / (2.0 * contagens[alvo])
        modelo = criar_classificador()
        modelo.fit(
            x_treino,
            alvo,
            categorical_feature=COLUNAS_CATEGORICAS,
            sample_weight=pesos,
        )
        return modelo.predict_proba(x_previsao)[:, 1]
    if nome in {"continuo_q95", "continuo_q99"}:
        quantil = 0.95 if nome.endswith("q95") else 0.99
        modelo = criar_regressor()
        modelo.fit(
            x_treino,
            alvos["ideal"],
            categorical_feature=COLUNAS_CATEGORICAS,
            sample_weight=normalizar_pesos(
                alvos["peso_continuo"], quantil
            ),
        )
        return modelo.predict(x_previsao)
    if nome == "utilidade":
        modelo = criar_classificador()
        modelo.fit(
            x_treino,
            alvos["utilidade"],
            categorical_feature=COLUNAS_CATEGORICAS,
            sample_weight=normalizar_pesos(
                alvos["peso_utilidade"], 0.99
            ),
        )
        return modelo.predict_proba(x_previsao)[:, 1]
    raise ValueError(nome)


def treinar_todos_scores(x_treino, y, base, especialista, x_previsao):
    alvos = construir_alvos(y, base, especialista)
    return {
        nome: treinar_score(nome, x_treino, alvos, x_previsao)
        for nome in NOMES_SCORES
    }


def gerar_scores_internos(x, y, base, especialista, semente):
    scores = {nome: np.full(len(x), np.nan) for nome in NOMES_SCORES}
    cv = KFold(n_splits=4, shuffle=True, random_state=semente)
    for it, iv in cv.split(x):
        previstos = treinar_todos_scores(
            x.iloc[it], y[it], base[it], especialista[it], x.iloc[iv]
        )
        for nome in NOMES_SCORES:
            scores[nome][iv] = previstos[nome]
    assert all(np.isfinite(valor).all() for valor in scores.values())
    return scores


def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def aplicar_estrategia(nome, parametros, scores, base):
    if nome.startswith("continuo_q99"):
        peso = np.clip(
            parametros["intercepto"]
            + parametros["inclinacao"] * scores["continuo_q99"],
            0.0,
            1.0,
        )
    elif nome.startswith("continuo_q95"):
        peso = np.clip(
            parametros["intercepto"]
            + parametros["inclinacao"] * scores["continuo_q95"],
            0.0,
            1.0,
        )
    elif nome.startswith("utilidade_gate_faixa"):
        peso = sigmoide(
            (scores["utilidade"] - parametros["centro_score"])
            / parametros["escala_score"]
        )
    elif nome == "faixa_pura":
        peso = np.full(len(base), parametros["intensidade"], dtype=float)
    elif nome == "preco_puro":
        peso = np.full(len(base), parametros["intensidade"], dtype=float)
    else:
        raise ValueError(nome)

    if "gate_faixa" in nome or nome == "faixa_pura":
        peso *= np.power(
            np.clip(scores["faixa_1m"], 0.0, 1.0), parametros["gamma"]
        )
    if "gate_preco" in nome or nome.endswith("_preco") or nome == "preco_puro":
        peso *= sigmoide(
            (base - parametros["centro_preco"])
            / parametros["largura_preco"]
        )
    return np.clip(peso, 0.0, 1.0)


def grade_parametros(nome, scores):
    afins = [
        {"intercepto": float(i), "inclinacao": float(s)}
        for i in np.arange(-0.40, 0.101, 0.10)
        for s in np.arange(0.50, 2.01, 0.25)
    ]
    gammas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    precos = [850_000, 900_000, 950_000, 1_000_000,
              1_050_000, 1_100_000, 1_200_000, 1_300_000,
              1_400_000, 1_500_000]
    larguras = [50_000, 100_000, 150_000, 250_000]
    if nome == "continuo_q99_livre":
        return afins
    if nome in {"continuo_q99_gate_faixa", "continuo_q95_gate_faixa"}:
        return [{**a, "gamma": g} for a in afins for g in gammas]
    if nome in {"continuo_q99_gate_preco", "continuo_q95_gate_preco"}:
        return [
            {**a, "centro_preco": c, "largura_preco": l}
            for a in afins for c in precos for l in larguras
        ]
    if nome == "continuo_q99_gate_faixa_preco":
        return [
            {
                **a,
                "gamma": g,
                "centro_preco": c,
                "largura_preco": l,
            }
            for a in afins
            for g in gammas
            for c in [1_000_000, 1_100_000, 1_200_000, 1_300_000, 1_400_000]
            for l in [100_000, 200_000, 300_000]
        ]
    if nome.startswith("utilidade_gate_faixa"):
        score = scores["utilidade"]
        centros = np.unique(np.quantile(score, np.linspace(0.20, 0.90, 8)))
        desvio = max(float(np.std(score)), 1e-6)
        base = [
            {
                "centro_score": float(c),
                "escala_score": float(desvio * e),
                "gamma": g,
            }
            for c in centros for e in [0.10, 0.25, 0.50, 1.0]
            for g in gammas
        ]
        if nome == "utilidade_gate_faixa":
            return base
        return [
            {
                **a,
                "centro_preco": c,
                "largura_preco": l,
            }
            for a in base
            for c in [1_000_000, 1_100_000, 1_200_000, 1_300_000, 1_400_000]
            for l in [100_000, 200_000, 300_000]
        ]
    if nome == "faixa_pura":
        return [
            {"intensidade": i, "gamma": g}
            for i in [0.25, 0.50, 0.75, 1.0]
            for g in gammas
        ]
    if nome == "preco_puro":
        return [
            {"intensidade": i, "centro_preco": c, "largura_preco": l}
            for i in [0.25, 0.50, 0.75, 1.0]
            for c in precos for l in larguras
        ]
    raise ValueError(nome)


def escolher_parametros(nome, y, base, especialista, scores):
    melhor = None
    for parametros in grade_parametros(nome, scores):
        peso = aplicar_estrategia(nome, parametros, scores, base)
        pred = base + peso * (especialista - base)
        erro = rmspe(y, pred)
        if melhor is None or erro < melhor["rmspe_treino"]:
            melhor = {**parametros, "rmspe_treino": float(erro)}
    return melhor


def avaliar_aninhado(x, meta):
    y = meta["preco"].to_numpy(dtype=float)
    base = meta["base"].to_numpy(dtype=float)
    especialista = meta["especialista_caro"].to_numpy(dtype=float)
    folds = meta["fold"].to_numpy(dtype=int)
    scores_oof = {nome: np.full(len(meta), np.nan) for nome in NOMES_SCORES}
    pesos_oof = {nome: np.full(len(meta), np.nan) for nome in ESTRATEGIAS}
    parametros_folds = []
    inicio = time.perf_counter()
    for fold in sorted(np.unique(folds)):
        it = np.flatnonzero(folds != fold)
        iv = np.flatnonzero(folds == fold)
        scores_inner = gerar_scores_internos(
            x.iloc[it], y[it], base[it], especialista[it], 100 + fold
        )
        scores_validacao = treinar_todos_scores(
            x.iloc[it], y[it], base[it], especialista[it], x.iloc[iv]
        )
        for nome in NOMES_SCORES:
            scores_oof[nome][iv] = scores_validacao[nome]
        for estrategia in ESTRATEGIAS:
            parametros = escolher_parametros(
                estrategia,
                y[it],
                base[it],
                especialista[it],
                scores_inner,
            )
            pesos_oof[estrategia][iv] = aplicar_estrategia(
                estrategia, parametros, scores_validacao, base[iv]
            )
            parametros_folds.append(
                {"estrategia": estrategia, "fold": int(fold), **parametros}
            )
        print(f"fold externo {fold}/5 concluido")
    assert all(np.isfinite(v).all() for v in scores_oof.values())
    assert all(np.isfinite(v).all() for v in pesos_oof.values())

    parametros_finais = {
        nome: escolher_parametros(nome, y, base, especialista, scores_oof)
        for nome in ESTRATEGIAS
    }
    resumos, saidas = [], {}
    for nome in ESTRATEGIAS:
        peso = pesos_oof[nome]
        pred = base + peso * (especialista - base)
        linha = {
            "estrategia": nome,
            "rmspe_crossfit_aninhado": rmspe(y, pred),
            "ganho_pp_vs_base": 100.0 * (rmspe(y, base) - rmspe(y, pred)),
            "peso_medio": float(peso.mean()),
            "peso_mediano": float(np.median(peso)),
            "p90_peso": float(np.quantile(peso, 0.90)),
            "fracao_peso_zero": float(np.mean(peso == 0.0)),
            "fracao_peso_acima_05": float(np.mean(peso > 0.50)),
        }
        for fold in sorted(np.unique(folds)):
            m = folds == fold
            linha[f"rmspe_fold_{fold}"] = rmspe(y[m], pred[m])
            linha[f"ganho_fold_{fold}_pp"] = 100.0 * (
                rmspe(y[m], base[m]) - rmspe(y[m], pred[m])
            )
        resumos.append(linha)
        saidas[nome] = {"peso": peso, "pred": pred}
    duracao = time.perf_counter() - inicio
    return (
        scores_oof,
        saidas,
        pd.DataFrame(resumos).sort_values("rmspe_crossfit_aninhado"),
        pd.DataFrame(parametros_folds),
        parametros_finais,
        duracao,
    )


def preparar_features(df, componentes, especialista):
    entrada = df if "preco" in df else df.assign(preco=np.nan)
    x = criar_features_catboost(entrada).drop(columns=["Id", "preco"])
    base = np.asarray(componentes["base"], dtype=float)
    esp = np.asarray(especialista, dtype=float)
    x["pred_pipeline_atual"] = base
    x["pred_especialista_caro"] = esp
    x["log_pred_pipeline_atual"] = np.log1p(base)
    x["log_pred_especialista_caro"] = np.log1p(esp)
    x["diferenca_caro_atual"] = esp - base
    x["razao_caro_atual"] = esp / base
    for coluna in [
        "generalista",
        "especialista_barato",
        "catboost_global",
        "blend_arvores",
        "peso_especialista_barato",
        "fracao_arvores",
        "tem_correspondencia",
    ]:
        x[coluna] = np.asarray(componentes[coluna])
    x["discordancia_cat_arvores"] = (
        x["catboost_global"] - x["blend_arvores"]
    ) / base
    x["amplitude_componentes"] = (
        np.maximum.reduce([
            base,
            esp,
            x["catboost_global"].to_numpy(),
            x["blend_arvores"].to_numpy(),
        ])
        - np.minimum.reduce([
            base,
            esp,
            x["catboost_global"].to_numpy(),
            x["blend_arvores"].to_numpy(),
        ])
    ) / base
    assert np.isfinite(x.select_dtypes(include=[np.number])).all().all()
    return x


def componentes_oof(pasta):
    atual = pd.read_csv(os.path.join(pasta, "juiz_componentes_oof.csv"))
    exp = pd.read_csv(
        os.path.join(pasta, "especialista_caros_estabilidade_oof.csv")
    )
    corr = pd.read_csv(os.path.join(pasta, "catboost_correspondencias_oof.csv"))
    assert np.array_equal(atual["Id"], exp["Id"])
    assert np.array_equal(atual["fold"], exp["fold_s42"])
    return atual, exp, {
        "base": atual["pred_ancorada_publico"],
        "generalista": atual["generalista"],
        "especialista_barato": atual["especialista"],
        "catboost_global": atual["catboost_global"],
        "blend_arvores": atual["blend_arvores"],
        "peso_especialista_barato": atual["peso_especialista"],
        "fracao_arvores": atual["r_arvores_ancorada_publico"],
        "tem_correspondencia": corr["previsao_correspondencia"].notna().astype(int),
    }


def componentes_holdout(raiz, df):
    pasta = os.path.join(raiz, "resultados")
    atual = pd.read_csv(os.path.join(pasta, "juiz_componentes_holdout.csv"))
    exp = pd.read_csv(os.path.join(pasta, "especialista_caros_holdout_id.csv"))
    cat = pd.read_csv(
        os.path.join(pasta, "catboost_correspondencias_holdout_id.csv")
    )
    assert np.array_equal(atual["Id"], exp["Id"])
    ids = set(atual["Id"])
    mascara = df["Id"].isin(ids).to_numpy()
    df_h = df.loc[mascara]
    assert np.array_equal(df_h["Id"], atual["Id"])
    pred_cat = aplicar_correspondencias(
        cat["catboost_alpha_1.50"].to_numpy(),
        cat["previsao_correspondencia"].to_numpy(),
        PESO_CORRESPONDENCIA,
    )
    arvores = (atual["generalista"].to_numpy() - 0.60 * pred_cat) / 0.40
    componentes = {
        "base": atual["pred_ancorada_publico"],
        "generalista": atual["generalista"],
        "especialista_barato": atual["especialista"],
        "catboost_global": pred_cat,
        "blend_arvores": arvores,
        "peso_especialista_barato": atual["peso_alvo_projetado_01"],
        "fracao_arvores": atual["r_arvores_ancorada_publico"],
        "tem_correspondencia": cat["previsao_correspondencia"].notna().astype(int),
    }
    x_h = preparar_features(df_h, componentes, exp["especialista_caros"])
    return atual, exp, x_h, mascara


def auditar_holdout(x_oof, meta, raiz, df):
    atual_h, exp_h, x_h, mascara_h = componentes_holdout(raiz, df)
    treino = ~mascara_h
    y_t = meta.loc[treino, "preco"].to_numpy(dtype=float)
    b_t = meta.loc[treino, "base"].to_numpy(dtype=float)
    e_t = meta.loc[treino, "especialista_caro"].to_numpy(dtype=float)
    x_t = x_oof.loc[treino].reset_index(drop=True)
    scores_inner = gerar_scores_internos(x_t, y_t, b_t, e_t, 4242)
    scores_h = treinar_todos_scores(
        x_t, y_t, b_t, e_t, x_h.reset_index(drop=True)
    )
    y_h = atual_h["preco"].to_numpy(dtype=float)
    b_h = atual_h["pred_ancorada_publico"].to_numpy(dtype=float)
    e_h = exp_h["especialista_caros"].to_numpy(dtype=float)
    resultados, detalhes = [], atual_h[["Id", "preco"]].copy()
    detalhes["base"] = b_h
    detalhes["especialista_caro"] = e_h
    parametros = {}
    for nome in ESTRATEGIAS:
        par = escolher_parametros(nome, y_t, b_t, e_t, scores_inner)
        peso = aplicar_estrategia(nome, par, scores_h, b_h)
        pred = b_h + peso * (e_h - b_h)
        parametros[nome] = par
        resultados.append({
            "estrategia": nome,
            "rmspe_holdout": rmspe(y_h, pred),
            "ganho_pp_vs_base": 100.0 * (rmspe(y_h, b_h) - rmspe(y_h, pred)),
            "peso_medio": float(peso.mean()),
            "peso_mediano": float(np.median(peso)),
            "p90_peso": float(np.quantile(peso, 0.90)),
        })
        detalhes[f"peso_{nome}"] = peso
        detalhes[f"pred_{nome}"] = pred
    return (
        pd.DataFrame(resultados).sort_values("rmspe_holdout"),
        detalhes,
        parametros,
    )


def diagnosticar_faixas(meta, saidas):
    y = meta["preco"].to_numpy(dtype=float)
    base = meta["base"].to_numpy(dtype=float)
    linhas = []
    for nome, dados in saidas.items():
        for inferior, superior in FAIXAS_DIAGNOSTICO:
            m = (y >= inferior) & (y < superior)
            linhas.append({
                "estrategia": nome,
                "limite_inferior": inferior,
                "limite_superior": superior,
                "quantidade": int(m.sum()),
                "rmspe_base": rmspe(y[m], base[m]),
                "rmspe_juiz": rmspe(y[m], dados["pred"][m]),
                "ganho_pp": 100.0 * (
                    rmspe(y[m], base[m]) - rmspe(y[m], dados["pred"][m])
                ),
                "peso_medio": float(dados["peso"][m].mean()),
            })
    return pd.DataFrame(linhas)


def executar():
    raiz, df = carregar_treino()
    pasta = os.path.join(raiz, "resultados")
    atual, exp, componentes = componentes_oof(pasta)
    assert np.array_equal(df["Id"], atual["Id"])
    x = preparar_features(
        df, componentes, exp["especialista_s42_calibrado_cf"]
    )
    meta = df[["Id", "preco"]].copy()
    meta["fold"] = atual["fold"]
    meta["base"] = atual["pred_ancorada_publico"]
    meta["especialista_caro"] = exp["especialista_s42_calibrado_cf"]

    scores, saidas, resumo, parametros_folds, parametros_finais, duracao = (
        avaliar_aninhado(x, meta)
    )
    holdout, detalhe_h, parametros_h = auditar_holdout(x, meta, raiz, df)
    faixas = diagnosticar_faixas(meta, saidas)

    oof = meta.copy()
    for nome, valor in scores.items():
        oof[f"score_{nome}"] = valor
    for nome, dados in saidas.items():
        oof[f"peso_{nome}"] = dados["peso"]
        oof[f"pred_{nome}"] = dados["pred"]
    oof.to_csv(os.path.join(pasta, "juiz_caros_oof.csv"), index=False)
    resumo["duracao_busca_segundos"] = duracao
    resumo.to_csv(os.path.join(pasta, "juiz_caros_resumo.csv"), index=False)
    parametros_folds.to_csv(
        os.path.join(pasta, "juiz_caros_parametros_folds.csv"), index=False
    )
    holdout.to_csv(
        os.path.join(pasta, "juiz_caros_holdout_resumo.csv"), index=False
    )
    detalhe_h.to_csv(
        os.path.join(pasta, "juiz_caros_holdout.csv"), index=False
    )
    faixas.to_csv(
        os.path.join(pasta, "juiz_caros_faixas.csv"), index=False
    )
    with open(
        os.path.join(pasta, "juiz_caros_parametros_finais.json"),
        "w", encoding="utf-8"
    ) as arquivo:
        json.dump(parametros_finais, arquivo, ensure_ascii=False, indent=2)

    base_global = rmspe(meta["preco"], meta["base"])
    base_h = rmspe(detalhe_h["preco"], detalhe_h["base"])
    candidatos = resumo.merge(holdout, on="estrategia", suffixes=("_oof", "_h"))
    candidatos["folds_melhores"] = candidatos.apply(
        lambda linha: sum(linha[f"ganho_fold_{f}_pp"] > 0 for f in range(1, 6)),
        axis=1,
    )
    candidatos = candidatos.sort_values("rmspe_crossfit_aninhado")
    elegiveis = candidatos[
        (candidatos["rmspe_crossfit_aninhado"] < base_global - 0.0003)
        & (candidatos["rmspe_holdout"] < base_h)
        & (candidatos["folds_melhores"] >= 4)
    ]
    vencedor = None if elegiveis.empty else elegiveis.iloc[0]
    decisao = {
        "rmspe_base_oof": base_global,
        "rmspe_base_holdout": base_h,
        "criterio_ganho_oof_minimo_pp": 0.03,
        "criterio_folds_melhores": 4,
        "estrategias_elegiveis": elegiveis["estrategia"].tolist(),
        "estrategia_escolhida": None if vencedor is None else vencedor["estrategia"],
        "rmspe_escolhido_oof": None if vencedor is None else float(
            vencedor["rmspe_crossfit_aninhado"]
        ),
        "rmspe_escolhido_holdout": None if vencedor is None else float(
            vencedor["rmspe_holdout"]
        ),
        "gerar_submission": vencedor is not None,
        "parametros_holdout": parametros_h,
    }
    with open(
        os.path.join(pasta, "juiz_caros_decisao_semente42.json"),
        "w", encoding="utf-8"
    ) as arquivo:
        json.dump(decisao, arquivo, ensure_ascii=False, indent=2)

    print("\nResumo OOF aninhado:")
    print(resumo.to_string(index=False))
    print("\nHoldout por ID:")
    print(holdout.to_string(index=False))
    print("\nDecisao:")
    print(json.dumps(decisao, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    executar()
