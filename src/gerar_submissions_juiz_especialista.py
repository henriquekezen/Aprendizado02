"""Audita o juiz escolhido e gera tres submissions do blend roteado.

O generalista publico fica congelado. O especialista e o CatBoost alpha 2
calibrado para imoveis baratos. O juiz estima se trocar o generalista pelo
especialista reduz a perda RMSPE por linha.
"""

import json
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from comparar_juizes_especialista import (
    criar_classificador,
    pesos_sigmoide,
    preparar_features,
    preparar_features_com_previsoes,
)
from gerar_oof_generalista_60_40 import treinar_arvores
from testar_catboost_alphas import (
    COLUNAS_CATEGORICAS,
    calcular_pesos,
    carregar_treino,
    corrigir_dados,
    separar_xy,
)
from testar_catboost_correspondencias import (
    aplicar_correspondencias,
    prever_correspondencias,
)


FATOR_ESPECIALISTA = 0.914489
LIMITE_BARATO = 355_000.0
PESO_CORRESPONDENCIA = 0.30
LAMBDAS_SUBMISSION = [0.40, 0.70, 1.00]
NOME_JUIZ = "classificador_utilidade"


def rmspe(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    return np.sqrt(np.mean(np.square((p - y) / y)))


def carregar_teste(raiz):
    caminho = os.path.join(raiz, "data", "conjunto_de_teste (3).csv")
    return corrigir_dados(pd.read_csv(caminho)).reset_index(drop=True)


def criar_especialista():
    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=600,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=5.0,
        random_strength=0.0,
        bootstrap_type="Bayesian",
        bagging_temperature=1.0,
        boosting_type="Ordered",
        random_seed=42,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )


def treinar_especialista(df_treino, df_previsao):
    x_treino, y_treino = separar_xy(df_treino)
    x_previsao, _ = separar_xy(
        df_previsao.assign(preco=np.nan)
        if "preco" not in df_previsao
        else df_previsao
    )
    modelo = criar_especialista()
    inicio = time.perf_counter()
    modelo.fit(
        x_treino,
        np.log1p(y_treino),
        cat_features=COLUNAS_CATEGORICAS,
        sample_weight=calcular_pesos(y_treino, 2.0),
    )
    previsao = np.expm1(modelo.predict(x_previsao)) * FATOR_ESPECIALISTA
    return previsao, time.perf_counter() - inicio


def alvo_e_pesos_juiz(meta):
    alvo = meta["delta_loss"].gt(0).astype(int)
    pesos = np.abs(meta["delta_loss"].to_numpy())
    pesos = np.clip(pesos, 1e-5, np.quantile(pesos, 0.99))
    pesos /= pesos.mean()
    return alvo, pesos


def treinar_juiz(x, meta, mascara=None):
    if mascara is None:
        mascara = np.ones(len(meta), dtype=bool)
    alvo, pesos = alvo_e_pesos_juiz(meta)
    modelo = criar_classificador()
    inicio = time.perf_counter()
    modelo.fit(
        x.loc[mascara],
        alvo.loc[mascara],
        cat_features=COLUNAS_CATEGORICAS,
        sample_weight=pesos[mascara],
    )
    return modelo, time.perf_counter() - inicio


def carregar_bases_oof(raiz, df):
    pasta = os.path.join(raiz, "resultados")
    geral = pd.read_csv(os.path.join(pasta, "generalista_60_40_oof.csv"))
    especialista = pd.read_csv(
        os.path.join(pasta, "catboost_baratos_curva_iteracoes_d7_rs0_oof.csv")
    )
    cat = pd.read_csv(os.path.join(pasta, "catboost_correspondencias_oof.csv"))
    meta = pd.read_csv(os.path.join(pasta, "juizes_especialista_oof.csv"))
    for base in [geral, especialista, cat, meta]:
        assert np.array_equal(df["Id"].to_numpy(), base["Id"].to_numpy())
    x = preparar_features(df, geral, especialista, cat)
    return geral, especialista, cat, meta, x


def carregar_parametros_mapeamento(raiz):
    caminho = os.path.join(
        raiz, "resultados", "juizes_especialista_parametros_finais.json"
    )
    with open(caminho, encoding="utf-8") as arquivo:
        return json.load(arquivo)[NOME_JUIZ]


def auditar_holdout(
    raiz,
    df,
    meta,
    x_oof,
    centro,
    escala,
):
    caminho = os.path.join(
        raiz, "resultados", "catboost_correspondencias_holdout_id.csv"
    )
    cat_holdout = pd.read_csv(caminho)
    ids_validacao = set(cat_holdout["Id"])
    mascara_validacao = df["Id"].isin(ids_validacao).to_numpy()
    mascara_treino = ~mascara_validacao
    treino = df.loc[mascara_treino]
    validacao = df.loc[mascara_validacao]
    assert np.array_equal(validacao["Id"].to_numpy(), cat_holdout["Id"].to_numpy())

    pred_xgb, pred_lgb, tempo_arvores = treinar_arvores(treino, validacao)
    correspondencia = prever_correspondencias(treino, validacao)
    pred_arvores = aplicar_correspondencias(
        0.50 * pred_xgb + 0.50 * pred_lgb,
        correspondencia,
        PESO_CORRESPONDENCIA,
    )
    pred_cat = aplicar_correspondencias(
        cat_holdout["catboost_alpha_1.50"].to_numpy(),
        cat_holdout["previsao_correspondencia"].to_numpy(),
        PESO_CORRESPONDENCIA,
    )
    pred_geral = 0.60 * pred_cat + 0.40 * pred_arvores
    pred_especialista, tempo_especialista = treinar_especialista(
        treino, validacao
    )
    x_validacao = preparar_features_com_previsoes(
        validacao,
        pred_geral,
        pred_especialista,
        pred_cat,
        pred_arvores,
        ~np.isnan(correspondencia),
    )
    juiz, tempo_juiz = treinar_juiz(x_oof, meta, mascara_treino)
    score = juiz.predict_proba(x_validacao)[:, 1]
    peso = pesos_sigmoide(score, centro, escala)
    y = validacao["preco"].to_numpy()

    linhas = []
    for intensidade in [0.0, *LAMBDAS_SUBMISSION]:
        pred = pred_geral + intensidade * peso * (
            pred_especialista - pred_geral
        )
        linhas.append(
            {
                "intensidade": intensidade,
                "rmspe": rmspe(y, pred),
                "rmspe_baratos_ate_355k": rmspe(
                    y[y <= LIMITE_BARATO], pred[y <= LIMITE_BARATO]
                ),
                "rmspe_caros_acima_355k": rmspe(
                    y[y > LIMITE_BARATO], pred[y > LIMITE_BARATO]
                ),
            }
        )
    detalhe = pd.DataFrame(
        {
            "Id": validacao["Id"].to_numpy(),
            "preco": y,
            "generalista": pred_geral,
            "especialista": pred_especialista,
            "score_juiz": score,
            "peso_juiz": peso,
        }
    )
    detalhe.to_csv(
        os.path.join(raiz, "resultados", "juiz_especialista_holdout_id.csv"),
        index=False,
    )
    resumo = pd.DataFrame(linhas)
    resumo["peso_medio_juiz"] = peso.mean()
    resumo["peso_mediano_juiz"] = np.median(peso)
    resumo["tempo_arvores_segundos"] = tempo_arvores
    resumo["tempo_especialista_segundos"] = tempo_especialista
    resumo["tempo_juiz_segundos"] = tempo_juiz
    resumo.to_csv(
        os.path.join(
            raiz, "resultados", "juiz_especialista_holdout_id_resumo.csv"
        ),
        index=False,
    )
    return resumo


def avaliar_intensidades_oof(meta, centro, escala):
    y = meta["preco"].to_numpy()
    geral = meta["generalista"].to_numpy()
    especialista = meta["especialista"].to_numpy()
    score = meta[f"score_{NOME_JUIZ}"].to_numpy()
    peso = pesos_sigmoide(score, centro, escala)
    linhas = []
    for intensidade in [0.0, *LAMBDAS_SUBMISSION]:
        pred = geral + intensidade * peso * (especialista - geral)
        linhas.append(
            {
                "intensidade": intensidade,
                "rmspe_oof": rmspe(y, pred),
                "peso_especialista_medio": intensidade * peso.mean(),
                "peso_especialista_p90": intensidade * np.quantile(peso, 0.90),
            }
        )
    return pd.DataFrame(linhas), peso


def preparar_features_teste(
    raiz,
    df_teste,
    pred_geral,
    pred_especialista,
):
    pasta_resultados = os.path.join(raiz, "resultados")
    pasta_submissions = os.path.join(raiz, "submissions")
    componentes_cat = pd.read_csv(
        os.path.join(
            pasta_resultados, "catboost_previsoes_teste_alphas_corr30.csv"
        )
    )
    arvores = pd.read_csv(
        os.path.join(
            pasta_submissions,
            "submission_blend_50_sample_weight_a150_corr30.csv",
        )
    )
    for base in [componentes_cat, arvores]:
        assert np.array_equal(df_teste["Id"].to_numpy(), base["Id"].to_numpy())
    pred_cat = componentes_cat["catboost_a150_corr30"].to_numpy()
    pred_arvores = arvores["preco"].to_numpy()
    x = preparar_features_com_previsoes(
        df_teste.assign(preco=np.nan),
        pred_geral,
        pred_especialista,
        pred_cat,
        pred_arvores,
        componentes_cat["previsao_correspondencia"].notna(),
    )
    return x, pred_cat, pred_arvores


def gerar_submissions(
    raiz,
    df,
    df_teste,
    meta,
    x_oof,
    centro,
    escala,
    resumo_holdout,
):
    pasta_submissions = os.path.join(raiz, "submissions")
    geral_teste = pd.read_csv(
        os.path.join(
            pasta_submissions,
            "submission_blend_global_cat60_xgb20_lgb20_a150_corr30.csv",
        )
    )
    modelo_resposta = pd.read_csv(
        os.path.join(raiz, "data", "exemplo_arquivo_respostas.csv")
    )
    for base in [geral_teste, modelo_resposta]:
        assert np.array_equal(df_teste["Id"].to_numpy(), base["Id"].to_numpy())

    pred_geral = geral_teste["preco"].to_numpy()
    pred_especialista, tempo_especialista = treinar_especialista(df, df_teste)
    x_teste, pred_cat, pred_arvores = preparar_features_teste(
        raiz, df_teste, pred_geral, pred_especialista
    )
    juiz, tempo_juiz = treinar_juiz(x_oof, meta)
    score = juiz.predict_proba(x_teste)[:, 1]
    peso = pesos_sigmoide(score, centro, escala)

    previsoes = df_teste[["Id"]].copy()
    previsoes["generalista"] = pred_geral
    previsoes["especialista"] = pred_especialista
    previsoes["catboost_global"] = pred_cat
    previsoes["blend_arvores"] = pred_arvores
    previsoes["score_juiz"] = score
    previsoes["peso_juiz"] = peso
    arquivos = []
    for intensidade in LAMBDAS_SUBMISSION:
        pred = pred_geral + intensidade * peso * (
            pred_especialista - pred_geral
        )
        coluna = f"previsao_lambda_{intensidade:.2f}"
        previsoes[coluna] = pred
        submission = modelo_resposta.copy()
        submission["preco"] = pred.round(2)
        nome = (
            "submission_juiz_utilidade_"
            f"lambda{int(round(intensidade * 100)):03d}.csv"
        )
        caminho = os.path.join(pasta_submissions, nome)
        submission.to_csv(caminho, index=False)
        assert submission.shape == (2000, 2)
        assert submission["Id"].is_unique
        assert not submission.isna().any().any()
        assert (submission["preco"] > 0).all()
        arquivos.append(nome)

    caminho_previsoes = os.path.join(
        raiz, "resultados", "juiz_especialista_previsoes_teste.csv"
    )
    previsoes.to_csv(caminho_previsoes, index=False)
    resumo_oof, peso_oof = avaliar_intensidades_oof(meta, centro, escala)
    resumo_oof.to_csv(
        os.path.join(
            raiz, "resultados", "juiz_especialista_intensidades_oof.csv"
        ),
        index=False,
    )
    relatorio = {
        "juiz": NOME_JUIZ,
        "centro_sigmoide": centro,
        "escala_sigmoide": escala,
        "fator_especialista": FATOR_ESPECIALISTA,
        "lambdas": LAMBDAS_SUBMISSION,
        "arquivos": arquivos,
        "peso_juiz_oof_media": float(peso_oof.mean()),
        "peso_juiz_teste_media": float(peso.mean()),
        "peso_juiz_teste_mediana": float(np.median(peso)),
        "peso_juiz_teste_p90": float(np.quantile(peso, 0.90)),
        "tempo_especialista_final_segundos": tempo_especialista,
        "tempo_juiz_final_segundos": tempo_juiz,
        "oof": resumo_oof.to_dict(orient="records"),
        "holdout_id": resumo_holdout.to_dict(orient="records"),
    }
    with open(
        os.path.join(
            raiz, "resultados", "juiz_especialista_submissions_resumo.json"
        ),
        "w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(relatorio, arquivo, ensure_ascii=False, indent=2)
    return arquivos, resumo_oof, previsoes, relatorio


def main():
    raiz, df = carregar_treino()
    df_teste = carregar_teste(raiz)
    _, _, _, meta, x_oof = carregar_bases_oof(raiz, df)
    parametros = carregar_parametros_mapeamento(raiz)
    centro = parametros["centro"]
    escala = parametros["escala"]

    print("Auditando o juiz no holdout por bloco de IDs...")
    resumo_holdout = auditar_holdout(
        raiz, df, meta, x_oof, centro, escala
    )
    print(resumo_holdout.to_string(index=False))

    print("\nTreinando especialista e juiz finais...")
    arquivos, resumo_oof, previsoes, relatorio = gerar_submissions(
        raiz,
        df,
        df_teste,
        meta,
        x_oof,
        centro,
        escala,
        resumo_holdout,
    )
    print("\nIntensidades OOF:")
    print(resumo_oof.to_string(index=False))
    print("\nDistribuicao do peso no teste:")
    print(
        f"media={previsoes['peso_juiz'].mean():.4f}, "
        f"mediana={previsoes['peso_juiz'].median():.4f}, "
        f"p90={previsoes['peso_juiz'].quantile(0.90):.4f}"
    )
    print("\nSubmissions:")
    for nome in arquivos:
        print(os.path.join(raiz, "submissions", nome))
    print(
        "Tempos finais: especialista="
        f"{relatorio['tempo_especialista_final_segundos']:.1f}s; "
        f"juiz={relatorio['tempo_juiz_final_segundos']:.1f}s"
    )


if __name__ == "__main__":
    main()
