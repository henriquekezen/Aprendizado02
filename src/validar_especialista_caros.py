"""Valida o especialista caro congelado em sementes, bootstrap e holdout."""

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from buscar_especialista_caros import (
    CONFIGURACAO_RAW_950,
    LIMITE_CAUDA,
    LIMITE_PRINCIPAL,
    LIMITES_CURVA,
    calcular_fator,
    carregar_treino,
    rmspe,
    treinar_modelo_arvore_fold,
)
from gerar_oof_generalista_60_40 import treinar_arvores
from refinar_especialista_caros import (
    aplicar_calibracao,
    aplicar_correspondencia,
    fit_calibracao,
)
from testar_catboost_correspondencias import prever_correspondencias


SEMENTES = [7, 42, 2026]
PESO_XGB = 0.725
CORR_XGB = 0.50
CORR_LGB = 0.50
CONFIG_XGB = {
    **CONFIGURACAO_RAW_950,
    "limite_treino": 900_000,
}
CONFIG_LGB = {
    **CONFIGURACAO_RAW_950,
    "limite_treino": 950_000,
}
LGB_OVERRIDES = {
    "num_leaves": 31,
    "max_depth": 7,
    "min_child_samples": 40,
    "n_estimators": 600,
}


def combinar(px, pl, corr):
    x = aplicar_correspondencia(px, corr, CORR_XGB)
    l = aplicar_correspondencia(pl, corr, CORR_LGB)
    return PESO_XGB * x + (1.0 - PESO_XGB) * l


def calibrar_log_afim_crossfit(y, p, folds):
    saida = np.full(len(y), np.nan)
    parametros = []
    for fold in sorted(np.unique(folds)):
        treino = (folds != fold) & (y >= LIMITE_PRINCIPAL)
        validacao = folds == fold
        par = fit_calibracao(y[treino], p[treino], "log_afim")
        saida[validacao] = aplicar_calibracao(p[validacao], "log_afim", par)
        parametros.append({"fold": int(fold), **par})
    final = fit_calibracao(
        y[y >= LIMITE_PRINCIPAL], p[y >= LIMITE_PRINCIPAL], "log_afim"
    )
    return saida, parametros, final


def gerar_oof(df, semente, imprimir=True):
    splits = list(KFold(n_splits=5, shuffle=True, random_state=semente).split(df))
    px = np.full(len(df), np.nan)
    pl = np.full(len(df), np.nan)
    corr = np.full(len(df), np.nan)
    folds = np.zeros(len(df), dtype=int)
    inicio = time.perf_counter()
    for fold, (it, iv) in enumerate(splits, start=1):
        treino, validacao = df.iloc[it], df.iloc[iv]
        px[iv] = treinar_modelo_arvore_fold(
            treino, validacao, CONFIG_XGB, "xgb"
        )
        pl[iv] = treinar_modelo_arvore_fold(
            treino, validacao, CONFIG_LGB, "lgb", overrides=LGB_OVERRIDES
        )
        corr[iv] = prever_correspondencias(treino, validacao)
        folds[iv] = fold
        if imprimir:
            print(f"semente={semente} fold={fold}/5 concluido")
    bruto = combinar(px, pl, corr)
    calibrado, parametros, final = calibrar_log_afim_crossfit(
        df["preco"].to_numpy(dtype=float), bruto, folds
    )
    return {
        "xgb": px,
        "lgb": pl,
        "correspondencia": corr,
        "bruto": bruto,
        "calibrado": calibrado,
        "fold": folds,
        "parametros": parametros,
        "parametro_final": final,
        "duracao": time.perf_counter() - inicio,
    }


def avaliar_semente(df, semente, dados):
    y = df["preco"].to_numpy(dtype=float)
    principal = y >= LIMITE_PRINCIPAL
    cauda = y >= LIMITE_CAUDA
    linha = {
        "semente": semente,
        "rmspe_950": rmspe(y[principal], dados["calibrado"][principal]),
        "rmspe_1300": rmspe(y[cauda], dados["calibrado"][cauda]),
        "duracao_segundos": dados["duracao"],
    }
    for fold in range(1, 6):
        mascara = (dados["fold"] == fold) & principal
        linha[f"rmspe_fold_{fold}"] = rmspe(
            y[mascara], dados["calibrado"][mascara]
        )
    return linha


def gerar_baseline_holdout(treino, validacao):
    inner = list(KFold(n_splits=5, shuffle=True, random_state=42).split(treino))
    p_inner = np.full(len(treino), np.nan)
    f_inner = np.zeros(len(treino), dtype=int)
    for fold, (it, iv) in enumerate(inner, start=1):
        tr, va = treino.iloc[it], treino.iloc[iv]
        px, pl, _ = treinar_arvores(tr, va)
        corr = prever_correspondencias(tr, va)
        bruto = 0.50 * px + 0.50 * pl
        p_inner[iv] = aplicar_correspondencia(bruto, corr, 0.30)
        f_inner[iv] = fold
    y_inner = treino["preco"].to_numpy(dtype=float)
    px, pl, _ = treinar_arvores(treino, validacao)
    corr = prever_correspondencias(treino, validacao)
    bruto = aplicar_correspondencia(0.50 * px + 0.50 * pl, corr, 0.30)
    # O fator final e aprendido apenas nas previsoes OOF do treino do holdout.
    mascara = y_inner >= LIMITE_PRINCIPAL
    fator = calcular_fator(y_inner[mascara], p_inner[mascara])
    return bruto * fator


def avaliar_holdout(raiz, df):
    quantidade = round(len(df) * 0.20)
    ids = set(df.nsmallest(quantidade, "Id")["Id"])
    treino = df[~df["Id"].isin(ids)]
    validacao = df[df["Id"].isin(ids)]
    dados_inner = gerar_oof(treino.reset_index(drop=True), 42, imprimir=False)
    parametro = dados_inner["parametro_final"]
    px = treinar_modelo_arvore_fold(treino, validacao, CONFIG_XGB, "xgb")
    pl = treinar_modelo_arvore_fold(
        treino, validacao, CONFIG_LGB, "lgb", overrides=LGB_OVERRIDES
    )
    corr = prever_correspondencias(treino, validacao)
    bruto = combinar(px, pl, corr)
    pred = aplicar_calibracao(bruto, "log_afim", parametro)
    baseline = gerar_baseline_holdout(treino, validacao)
    atual = pd.read_csv(
        os.path.join(raiz, "resultados", "juiz_componentes_holdout.csv")
    )
    assert np.array_equal(
        validacao["Id"].to_numpy(), atual["Id"].to_numpy()
    )
    pred_atual = atual["pred_ancorada_publico"].to_numpy(dtype=float)
    y = validacao["preco"].to_numpy(dtype=float)
    principal = y >= LIMITE_PRINCIPAL
    cauda = y >= LIMITE_CAUDA
    resumo = {
        "quantidade_total": len(validacao),
        "quantidade_950": int(principal.sum()),
        "quantidade_1300": int(cauda.sum()),
        "rmspe_especialista_950": rmspe(y[principal], pred[principal]),
        "rmspe_baseline_arvores_calibrado_950": rmspe(
            y[principal], baseline[principal]
        ),
        "rmspe_pipeline_atual_21_19_950": rmspe(
            y[principal], pred_atual[principal]
        ),
        "rmspe_especialista_1300": rmspe(y[cauda], pred[cauda]),
        "rmspe_baseline_arvores_calibrado_1300": rmspe(
            y[cauda], baseline[cauda]
        ),
        "rmspe_pipeline_atual_21_19_1300": rmspe(
            y[cauda], pred_atual[cauda]
        ),
        "parametros_calibracao": parametro,
    }
    detalhe = validacao[["Id", "preco"]].copy()
    detalhe["especialista_caros"] = pred
    detalhe["baseline_arvores_calibrado"] = baseline
    detalhe["pipeline_atual_21_19"] = pred_atual
    return resumo, detalhe


def bootstrap_pareado(y, especialista, baseline, limite, n=5000):
    mascara = np.asarray(y) >= limite
    y = np.asarray(y)[mascara]
    e = np.asarray(especialista)[mascara]
    b = np.asarray(baseline)[mascara]
    rng = np.random.default_rng(42)
    ganhos = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        ganhos[i] = rmspe(y[idx], b[idx]) - rmspe(y[idx], e[idx])
    return {
        "ganho_mediano": float(np.median(ganhos)),
        "ic95_inferior": float(np.quantile(ganhos, 0.025)),
        "ic95_superior": float(np.quantile(ganhos, 0.975)),
        "probabilidade_ganho": float(np.mean(ganhos > 0)),
    }


def comparar_faixas(y, especialistas, pipeline_atual, pipeline_calibrado):
    limites = [
        (0, 740_000),
        (740_000, 830_000),
        (830_000, 875_000),
        (875_000, 900_000),
        (900_000, 925_000),
        (925_000, 950_000),
        (950_000, 1_000_000),
        (1_000_000, 1_300_000),
        (1_300_000, 1_500_000),
        (1_500_000, 2_000_000),
        (2_000_000, np.inf),
    ]
    linhas = []
    for inferior, superior in limites:
        mascara = (y >= inferior) & (y < superior)
        erro_atual = np.abs((pipeline_atual[mascara] - y[mascara]) / y[mascara])
        scores = {
            semente: rmspe(y[mascara], pred[mascara])
            for semente, pred in especialistas.items()
        }
        re = float(np.mean(list(scores.values())))
        re_std = float(np.std(list(scores.values()), ddof=1))
        ra = rmspe(y[mascara], pipeline_atual[mascara])
        rc = rmspe(y[mascara], pipeline_calibrado[mascara])
        vitorias = []
        for pred in especialistas.values():
            erro_especialista = np.abs(
                (pred[mascara] - y[mascara]) / y[mascara]
            )
            vitorias.append(np.mean(erro_especialista < erro_atual))
        linhas.append({
            "limite_inferior": inferior,
            "limite_superior": superior,
            "quantidade": int(mascara.sum()),
            "rmspe_especialista_media_sementes": re,
            "desvio_especialista_sementes": re_std,
            **{
                f"rmspe_especialista_s{semente}": score
                for semente, score in scores.items()
            },
            "rmspe_pipeline_atual": ra,
            "rmspe_pipeline_atual_calibrado_caros": rc,
            "ganho_pp_versus_pipeline_atual": 100.0 * (ra - re),
            "ganho_pp_versus_pipeline_calibrado": 100.0 * (rc - re),
            "proporcao_linhas_especialista_melhor": float(
                np.mean(vitorias)
            ),
        })
    return pd.DataFrame(linhas)


def executar():
    raiz, df = carregar_treino()
    pasta = os.path.join(raiz, "resultados")
    resumos, dados_sementes = [], {}
    oof = df[["Id", "preco"]].copy()
    for semente in SEMENTES:
        dados = gerar_oof(df, semente)
        dados_sementes[semente] = dados
        resumos.append(avaliar_semente(df, semente, dados))
        oof[f"fold_s{semente}"] = dados["fold"]
        oof[f"especialista_s{semente}_bruto"] = dados["bruto"]
        oof[f"especialista_s{semente}_calibrado_cf"] = dados["calibrado"]

    holdout_resumo, holdout_detalhe = avaliar_holdout(raiz, df)
    rodada1 = pd.read_csv(
        os.path.join(pasta, "especialista_caros_rodada1_oof.csv")
    )
    baseline_arvores = rodada1[
        "baseline_arvores_corr30_calibrado_cf"
    ].to_numpy()
    baseline_atual = rodada1[
        "baseline_atual_21_19_calibrado_cf"
    ].to_numpy()
    pipeline_atual_bruto = rodada1[
        "baseline_atual_21_19_bruto"
    ].to_numpy()
    y = df["preco"].to_numpy(dtype=float)
    boot = {
        "versus_arvores_corr30": {
            "limite_950": bootstrap_pareado(
                y, dados_sementes[42]["calibrado"], baseline_arvores,
                LIMITE_PRINCIPAL
            ),
            "limite_1300": bootstrap_pareado(
                y, dados_sementes[42]["calibrado"], baseline_arvores,
                LIMITE_CAUDA
            ),
        },
        "versus_pipeline_atual_21_19": {
            "limite_950": bootstrap_pareado(
                y, dados_sementes[42]["calibrado"], baseline_atual,
                LIMITE_PRINCIPAL
            ),
            "limite_1300": bootstrap_pareado(
                y, dados_sementes[42]["calibrado"], baseline_atual,
                LIMITE_CAUDA
            ),
        },
    }
    curvas = []
    for semente, dados in dados_sementes.items():
        for limite in LIMITES_CURVA:
            mascara = y >= limite
            curvas.append({
                "semente": semente,
                "limite_preco": limite,
                "quantidade": int(mascara.sum()),
                "rmspe": rmspe(y[mascara], dados["calibrado"][mascara]),
            })

    pd.DataFrame(resumos).to_csv(
        os.path.join(pasta, "especialista_caros_estabilidade_sementes.csv"),
        index=False,
    )
    pd.DataFrame(curvas).to_csv(
        os.path.join(pasta, "especialista_caros_estabilidade_curvas.csv"),
        index=False,
    )
    comparar_faixas(
        y,
        {
            semente: dados["calibrado"]
            for semente, dados in dados_sementes.items()
        },
        pipeline_atual_bruto,
        baseline_atual,
    ).to_csv(
        os.path.join(pasta, "especialista_caros_comparacao_faixas.csv"),
        index=False,
    )
    oof.to_csv(
        os.path.join(pasta, "especialista_caros_estabilidade_oof.csv"),
        index=False,
    )
    holdout_detalhe.to_csv(
        os.path.join(pasta, "especialista_caros_holdout_id.csv"), index=False
    )
    consolidado = {
        "configuracao": {
            "limite_xgb": 900_000,
            "limite_lgb": 950_000,
            "target": "preco_bruto_dividido_1e6",
            "sample_weight": "1/preco**2 normalizado",
            "peso_xgb": PESO_XGB,
            "peso_lgb": 1.0 - PESO_XGB,
            "correspondencia_xgb": CORR_XGB,
            "correspondencia_lgb": CORR_LGB,
            "calibracao": "log_afim",
        },
        "sementes": resumos,
        "holdout": holdout_resumo,
        "bootstrap": boot,
    }
    with open(
        os.path.join(pasta, "especialista_caros_validacao_final.json"),
        "w", encoding="utf-8"
    ) as arquivo:
        json.dump(consolidado, arquivo, ensure_ascii=False, indent=2)
    print(pd.DataFrame(resumos).to_string(index=False))
    print("\nHoldout:")
    print(json.dumps(holdout_resumo, ensure_ascii=False, indent=2))
    print("\nBootstrap:")
    print(json.dumps(boot, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    executar()
