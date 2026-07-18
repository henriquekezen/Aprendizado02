"""Busca iterativa de modelos especializados em imoveis caros.

Rodada inicial: compara pesos por preco e treino restrito para XGBoost,
LightGBM, blends e CatBoost nos mesmos cinco folds do pipeline principal.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from pre_processamento import (
    criar_features_modelo_bairro_categorico,
    selecionar_bairros_frequentes,
)
from testar_catboost_alphas import (
    COLUNAS_CATEGORICAS,
    carregar_treino,
    criar_features_catboost,
)


LIMITE_PRINCIPAL = 950_000.0
LIMITE_CAUDA = 1_300_000.0
LIMITES_CURVA = [600_000, 740_000, 830_000, 950_000, 1_000_000,
                 1_300_000, 1_500_000, 2_000_000]

CONFIGURACOES_ARVORES_RODADA1 = [
    {"nome": "log_all_bm05", "target": "log", "beta": -0.5},
    {"nome": "log_all_b000", "target": "log", "beta": 0.0},
    {"nome": "log_all_b050", "target": "log", "beta": 0.5},
    {"nome": "log_all_b100", "target": "log", "beta": 1.0},
    {"nome": "log_sub830", "target": "log", "limite_treino": 830_000},
    {"nome": "log_sub950", "target": "log", "limite_treino": 950_000},
    {"nome": "raw_sub830_rmspe", "target": "raw", "limite_treino": 830_000,
     "peso_rmspe_raw": True},
    {"nome": "raw_sub950_rmspe", "target": "raw", "limite_treino": 950_000,
     "peso_rmspe_raw": True},
]

CONFIGURACOES_CAT_RODADA1 = [
    {"nome": "cat_log_all_b000", "target": "log", "beta": 0.0},
    {"nome": "cat_log_all_b050", "target": "log", "beta": 0.5},
    {"nome": "cat_log_all_b100", "target": "log", "beta": 1.0},
    {"nome": "cat_log_all_b150", "target": "log", "beta": 1.5},
    {"nome": "cat_log_sub830", "target": "log", "limite_treino": 830_000},
    {"nome": "cat_log_sub950", "target": "log", "limite_treino": 950_000},
]

CONFIGURACAO_RAW_950 = {
    "target": "raw",
    "limite_treino": 950_000,
    "peso_rmspe_raw": True,
}

CONFIGURACOES_RODADA2 = [
    # Fronteira de treino.
    {"nome": "xgb_thr900", "modelo": "xgb", "limite_treino": 900_000},
    {"nome": "xgb_thr950", "modelo": "xgb", "limite_treino": 950_000},
    {"nome": "xgb_thr1000", "modelo": "xgb", "limite_treino": 1_000_000},
    {"nome": "xgb_thr1100", "modelo": "xgb", "limite_treino": 1_100_000},
    # Capacidade XGBoost.
    {"nome": "xgb_d3_mc3_i1000", "modelo": "xgb",
     "overrides": {"max_depth": 3, "min_child_weight": 3, "n_estimators": 1000}},
    {"nome": "xgb_d3_mc5_i1000", "modelo": "xgb",
     "overrides": {"max_depth": 3, "min_child_weight": 5, "n_estimators": 1000}},
    {"nome": "xgb_d4_mc3_i1000", "modelo": "xgb",
     "overrides": {"max_depth": 4, "min_child_weight": 3, "n_estimators": 1000}},
    {"nome": "xgb_d4_mc10_i1000", "modelo": "xgb",
     "overrides": {"max_depth": 4, "min_child_weight": 10, "n_estimators": 1000}},
    {"nome": "xgb_d5_mc5_i1000", "modelo": "xgb",
     "overrides": {"max_depth": 5, "min_child_weight": 5, "n_estimators": 1000}},
    {"nome": "xgb_d5_mc10_i1000", "modelo": "xgb",
     "overrides": {"max_depth": 5, "min_child_weight": 10, "n_estimators": 1000}},
    {"nome": "xgb_d4_mc5_i1200", "modelo": "xgb",
     "overrides": {"n_estimators": 1200}},
    # Features XGBoost.
    {"nome": "xgb_features_caras", "modelo": "xgb", "features_caras": True},
    {"nome": "xgb_target_encoding", "modelo": "xgb", "target_encoding": True},
    {"nome": "xgb_features_caras_te", "modelo": "xgb",
     "features_caras": True, "target_encoding": True},
    # Fronteira e capacidade LightGBM.
    {"nome": "lgb_thr900", "modelo": "lgb", "limite_treino": 900_000},
    {"nome": "lgb_thr950", "modelo": "lgb", "limite_treino": 950_000},
    {"nome": "lgb_thr1000", "modelo": "lgb", "limite_treino": 1_000_000},
    {"nome": "lgb_thr1100", "modelo": "lgb", "limite_treino": 1_100_000},
    {"nome": "lgb_l15_d5_mc40_i600", "modelo": "lgb",
     "overrides": {"min_child_samples": 40, "n_estimators": 600}},
    {"nome": "lgb_l31_d7_mc20_i600", "modelo": "lgb",
     "overrides": {"num_leaves": 31, "max_depth": 7, "n_estimators": 600}},
    {"nome": "lgb_l31_d7_mc40_i600", "modelo": "lgb",
     "overrides": {"num_leaves": 31, "max_depth": 7,
                   "min_child_samples": 40, "n_estimators": 600}},
    {"nome": "lgb_l63_d8_mc20_i600", "modelo": "lgb",
     "overrides": {"num_leaves": 63, "max_depth": 8, "n_estimators": 600}},
    {"nome": "lgb_l63_d8_mc40_i800", "modelo": "lgb",
     "overrides": {"num_leaves": 63, "max_depth": 8,
                   "min_child_samples": 40, "n_estimators": 800}},
    {"nome": "lgb_features_caras", "modelo": "lgb", "features_caras": True},
    {"nome": "lgb_target_encoding", "modelo": "lgb", "target_encoding": True},
    {"nome": "lgb_features_caras_te", "modelo": "lgb",
     "features_caras": True, "target_encoding": True},
]

CONFIGURACOES_RODADA2B = [
    {"nome": "xgb_thr850", "modelo": "xgb", "limite_treino": 850_000},
    {"nome": "xgb_thr875", "modelo": "xgb", "limite_treino": 875_000},
    {"nome": "xgb_thr925", "modelo": "xgb", "limite_treino": 925_000},
    {"nome": "xgb900_d3_mc3_i1000", "modelo": "xgb", "limite_treino": 900_000,
     "overrides": {"max_depth": 3, "min_child_weight": 3, "n_estimators": 1000}},
    {"nome": "xgb900_d4_mc3_i1000", "modelo": "xgb", "limite_treino": 900_000,
     "overrides": {"max_depth": 4, "min_child_weight": 3, "n_estimators": 1000}},
    {"nome": "xgb900_d4_mc10_i1000", "modelo": "xgb", "limite_treino": 900_000,
     "overrides": {"max_depth": 4, "min_child_weight": 10, "n_estimators": 1000}},
    {"nome": "xgb900_d5_mc5_i1000", "modelo": "xgb", "limite_treino": 900_000,
     "overrides": {"max_depth": 5, "min_child_weight": 5, "n_estimators": 1000}},
    {"nome": "xgb900_i1200", "modelo": "xgb", "limite_treino": 900_000,
     "overrides": {"n_estimators": 1200}},
    {"nome": "xgb900_features_caras", "modelo": "xgb", "limite_treino": 900_000,
     "features_caras": True},
    {"nome": "xgb900_target_encoding", "modelo": "xgb", "limite_treino": 900_000,
     "target_encoding": True},
    {"nome": "xgb900_features_caras_te", "modelo": "xgb", "limite_treino": 900_000,
     "features_caras": True, "target_encoding": True},
    {"nome": "lgb900_l31_d7_mc40_i600", "modelo": "lgb", "limite_treino": 900_000,
     "overrides": {"num_leaves": 31, "max_depth": 7,
                   "min_child_samples": 40, "n_estimators": 600}},
    {"nome": "lgb900_l63_d8_mc40_i800", "modelo": "lgb", "limite_treino": 900_000,
     "overrides": {"num_leaves": 63, "max_depth": 8,
                   "min_child_samples": 40, "n_estimators": 800}},
    {"nome": "lgb900_features_caras_te", "modelo": "lgb", "limite_treino": 900_000,
     "features_caras": True, "target_encoding": True},
]


def rmspe(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return np.sqrt(np.mean(np.square((p - y) / y)))


def calcular_fator(y, p):
    razao = np.asarray(p, dtype=float) / np.asarray(y, dtype=float)
    return float(razao.sum() / np.square(razao).sum())


def calibrar_crossfit(y, p, folds, limite=LIMITE_PRINCIPAL):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    folds = np.asarray(folds)
    calibrada = p.copy()
    fatores = {}
    for fold in sorted(np.unique(folds)):
        ajuste = (folds != fold) & (y >= limite)
        fator = calcular_fator(y[ajuste], p[ajuste])
        fatores[int(fold)] = fator
        calibrada[folds == fold] *= fator
    return calibrada, fatores


def pesos_preco(y, beta):
    if beta == 0:
        return None
    y = np.asarray(y, dtype=float)
    pesos = np.power(y / np.median(y), beta)
    return pesos / pesos.mean()


def preparar_alvo(y, target):
    if target == "log":
        return np.log1p(y)
    if target == "raw":
        return np.asarray(y, dtype=float) / 1_000_000.0
    raise ValueError(target)


def restaurar_alvo(p, target):
    if target == "log":
        return np.expm1(p)
    if target == "raw":
        return np.asarray(p, dtype=float) * 1_000_000.0
    raise ValueError(target)


def mascara_e_pesos(y, configuracao):
    y = np.asarray(y, dtype=float)
    limite = configuracao.get("limite_treino")
    mascara = np.ones(len(y), dtype=bool) if limite is None else y >= limite
    y_usado = y[mascara]
    if configuracao.get("peso_rmspe_raw", False):
        pesos = np.power(y_usado, -2.0)
        pesos /= pesos.mean()
    else:
        pesos = pesos_preco(y_usado, configuracao.get("beta", 0.0))
    return mascara, pesos


def adicionar_features_caras(df):
    df = df.copy()
    df["log_area_util"] = np.log1p(df["area_util"].clip(lower=0))
    df["log_area_total"] = np.log1p(df["area_total"].clip(lower=0))
    df["razao_area_extra"] = df["area_extra"] / df["area_total"].replace(0, np.nan)
    df["suites_por_quarto"] = df["suites"] / df["quartos"].replace(0, np.nan)
    df["vagas_por_quarto"] = df["vagas"] / df["quartos"].replace(0, np.nan)
    df["area_x_vagas"] = df["area_util"] * df["vagas"]
    for coluna in ["razao_area_extra", "suites_por_quarto", "vagas_por_quarto"]:
        df[coluna] = df[coluna].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def _mapear_codificacao(df_referencia, df_destino, grupos, alvo, suavizacao=20.0):
    referencia = df_referencia[list(grupos)].copy()
    referencia["_alvo_te"] = np.asarray(alvo, dtype=float)
    global_alvo = float(np.mean(alvo))
    tabela = (
        referencia.groupby(list(grupos), dropna=False)["_alvo_te"]
        .agg(["mean", "count"])
        .reset_index()
    )
    tabela["_te"] = (
        tabela["mean"] * tabela["count"] + global_alvo * suavizacao
    ) / (tabela["count"] + suavizacao)
    destino = df_destino[list(grupos)].copy()
    destino["_ordem"] = np.arange(len(destino))
    valores = (
        destino.merge(tabela[list(grupos) + ["_te"]], on=list(grupos), how="left")
        .sort_values("_ordem")["_te"]
        .fillna(global_alvo)
        .to_numpy()
    )
    return valores


def adicionar_target_encodings(df_treino, df_validacao, treino, validacao):
    treino = treino.copy()
    validacao = validacao.copy()
    alvos = {
        "te_bairro_log_preco": (
            ["bairro"],
            np.log1p(df_treino["preco"].to_numpy(dtype=float)),
        ),
        "te_bairro_tipo_log_preco": (
            ["bairro", "tipo"],
            np.log1p(df_treino["preco"].to_numpy(dtype=float)),
        ),
        "te_bairro_log_preco_m2": (
            ["bairro"],
            np.log1p(
                df_treino["preco"].to_numpy(dtype=float)
                / df_treino["area_util"].clip(lower=1).to_numpy(dtype=float)
            ),
        ),
    }
    inner = list(KFold(n_splits=4, shuffle=True, random_state=42).split(df_treino))
    for nome, (grupos, alvo) in alvos.items():
        valores_treino = np.full(len(df_treino), np.nan)
        for indices_ajuste, indices_validacao in inner:
            valores_treino[indices_validacao] = _mapear_codificacao(
                df_treino.iloc[indices_ajuste],
                df_treino.iloc[indices_validacao],
                grupos,
                alvo[indices_ajuste],
            )
        valores_validacao = _mapear_codificacao(
            df_treino, df_validacao, grupos, alvo
        )
        assert np.isfinite(valores_treino).all()
        treino[nome] = valores_treino
        validacao[nome] = valores_validacao
    return treino, validacao


def preparar_arvores(df_treino, df_validacao, features_caras=False,
                     target_encoding=False):
    bairros = selecionar_bairros_frequentes(df_treino, minimo_imoveis=10)
    treino = criar_features_modelo_bairro_categorico(df_treino, bairros)
    validacao = criar_features_modelo_bairro_categorico(df_validacao, bairros)
    if features_caras:
        treino = adicionar_features_caras(treino)
        validacao = adicionar_features_caras(validacao)
    if target_encoding:
        treino, validacao = adicionar_target_encodings(
            df_treino, df_validacao, treino, validacao
        )
    return treino, validacao


def criar_xgb(**overrides):
    parametros = {
        "objective": "reg:squarederror",
        "n_estimators": 800,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "device": "cpu",
        "enable_categorical": True,
        "max_cat_to_onehot": 4,
        "random_state": 42,
        "n_jobs": -1,
    }
    parametros.update(overrides)
    return XGBRegressor(**parametros)


def criar_lgb(**overrides):
    parametros = {
        "objective": "regression",
        "n_estimators": 400,
        "learning_rate": 0.06,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 20,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "cat_smooth": 10.0,
        "cat_l2": 10.0,
        "min_data_per_group": 100,
        "max_cat_threshold": 32,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
    parametros.update(overrides)
    return LGBMRegressor(**parametros)


def criar_cat(**overrides):
    parametros = {
        "loss_function": "RMSE",
        "iterations": 800,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 5.0,
        "random_strength": 1.0,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 1.0,
        "boosting_type": "Ordered",
        "random_seed": 42,
        "allow_writing_files": False,
        "verbose": False,
        "thread_count": -1,
    }
    parametros.update(overrides)
    return CatBoostRegressor(**parametros)


def treinar_arvores_fold(df_treino, df_validacao, configuracao,
                         xgb_overrides=None, lgb_overrides=None,
                         features_caras=False):
    treino, validacao = preparar_arvores(df_treino, df_validacao, features_caras)
    x_treino = treino.drop(columns=["Id", "preco"])
    x_validacao = validacao.drop(columns=["Id", "preco"], errors="ignore")
    y = treino["preco"].to_numpy(dtype=float)
    mascara, pesos = mascara_e_pesos(y, configuracao)
    alvo = preparar_alvo(y[mascara], configuracao["target"])
    xgb = criar_xgb(**(xgb_overrides or {}))
    lgb = criar_lgb(**(lgb_overrides or {}))
    xgb.fit(x_treino.loc[mascara], alvo, sample_weight=pesos)
    lgb.fit(
        x_treino.loc[mascara],
        alvo,
        sample_weight=pesos,
        categorical_feature=["bairro"],
    )
    return (
        restaurar_alvo(xgb.predict(x_validacao), configuracao["target"]),
        restaurar_alvo(lgb.predict(x_validacao), configuracao["target"]),
    )


def treinar_modelo_arvore_fold(df_treino, df_validacao, configuracao,
                               modelo_nome, overrides=None,
                               features_caras=False, target_encoding=False):
    treino, validacao = preparar_arvores(
        df_treino, df_validacao, features_caras, target_encoding
    )
    x_treino = treino.drop(columns=["Id", "preco"])
    x_validacao = validacao.drop(columns=["Id", "preco"], errors="ignore")
    y = treino["preco"].to_numpy(dtype=float)
    mascara, pesos = mascara_e_pesos(y, configuracao)
    alvo = preparar_alvo(y[mascara], configuracao["target"])
    if modelo_nome == "xgb":
        modelo = criar_xgb(**(overrides or {}))
        modelo.fit(x_treino.loc[mascara], alvo, sample_weight=pesos)
    elif modelo_nome == "lgb":
        modelo = criar_lgb(**(overrides or {}))
        modelo.fit(
            x_treino.loc[mascara], alvo, sample_weight=pesos,
            categorical_feature=["bairro"]
        )
    else:
        raise ValueError(modelo_nome)
    return restaurar_alvo(modelo.predict(x_validacao), configuracao["target"])


def preparar_cat(df):
    modelo = adicionar_features_caras(criar_features_catboost(df))
    return modelo.drop(columns=["Id", "preco"], errors="ignore")


def treinar_cat_fold(df_treino, df_validacao, configuracao, overrides=None):
    x_treino = preparar_cat(df_treino)
    x_validacao = preparar_cat(df_validacao)
    y = df_treino["preco"].to_numpy(dtype=float)
    mascara, pesos = mascara_e_pesos(y, configuracao)
    modelo = criar_cat(**(overrides or {}))
    modelo.fit(
        x_treino.loc[mascara],
        preparar_alvo(y[mascara], configuracao["target"]),
        cat_features=COLUNAS_CATEGORICAS,
        sample_weight=pesos,
    )
    return restaurar_alvo(modelo.predict(x_validacao), configuracao["target"])


def avaliar_previsao(nome, familia, y, p, folds):
    p = np.maximum(np.asarray(p, dtype=float), 1.0)
    calibrada, fatores = calibrar_crossfit(y, p, folds)
    principal = y >= LIMITE_PRINCIPAL
    cauda = y >= LIMITE_CAUDA
    linha = {
        "nome": nome,
        "familia": familia,
        "rmspe_global_bruto": rmspe(y, p),
        "rmspe_950_bruto": rmspe(y[principal], p[principal]),
        "rmspe_950_calibrado_cf": rmspe(y[principal], calibrada[principal]),
        "rmspe_1300_bruto": rmspe(y[cauda], p[cauda]),
        "rmspe_1300_calibrado_cf": rmspe(y[cauda], calibrada[cauda]),
        "fator_final_950": calcular_fator(y[principal], p[principal]),
        "fator_min_fold": min(fatores.values()),
        "fator_max_fold": max(fatores.values()),
        "media_razao_950_bruto": float(np.mean(p[principal] / y[principal])),
    }
    for fold in sorted(np.unique(folds)):
        mascara = (folds == fold) & principal
        linha[f"rmspe_fold_{int(fold)}"] = rmspe(y[mascara], calibrada[mascara])
    curvas = []
    for limite in LIMITES_CURVA:
        mascara = y >= limite
        curvas.append({
            "nome": nome,
            "familia": familia,
            "limite_preco": limite,
            "quantidade": int(mascara.sum()),
            "rmspe_bruto": rmspe(y[mascara], p[mascara]),
            "rmspe_calibrado_cf": rmspe(y[mascara], calibrada[mascara]),
        })
    return linha, curvas, calibrada


def executar_rodada1():
    raiz, df = carregar_treino()
    splits = list(KFold(n_splits=5, shuffle=True, random_state=42).split(df))
    folds = np.zeros(len(df), dtype=int)
    for numero, (_, validacao) in enumerate(splits, start=1):
        folds[validacao] = numero
    y = df["preco"].to_numpy(dtype=float)
    previsoes = {}
    duracoes = {}

    for config in CONFIGURACOES_ARVORES_RODADA1:
        nome = config["nome"]
        px = np.full(len(df), np.nan)
        pl = np.full(len(df), np.nan)
        inicio = time.perf_counter()
        for numero, (it, iv) in enumerate(splits, start=1):
            pred_x, pred_l = treinar_arvores_fold(df.iloc[it], df.iloc[iv], config)
            px[iv] = pred_x
            pl[iv] = pred_l
            print(f"{nome}: fold {numero}/5 concluido")
        duracoes[nome] = time.perf_counter() - inicio
        previsoes[f"xgb_{nome}"] = px
        previsoes[f"lgb_{nome}"] = pl
        previsoes[f"blend50_{nome}"] = 0.50 * px + 0.50 * pl

    for config in CONFIGURACOES_CAT_RODADA1:
        nome = config["nome"]
        pc = np.full(len(df), np.nan)
        inicio = time.perf_counter()
        for numero, (it, iv) in enumerate(splits, start=1):
            pc[iv] = treinar_cat_fold(df.iloc[it], df.iloc[iv], config)
            print(f"{nome}: fold {numero}/5 concluido")
        duracoes[nome] = time.perf_counter() - inicio
        previsoes[nome] = pc

    geral = pd.read_csv(os.path.join(raiz, "resultados", "generalista_60_40_oof.csv"))
    juiz = pd.read_csv(os.path.join(raiz, "resultados", "juiz_componentes_oof.csv"))
    assert np.array_equal(df["Id"].to_numpy(), geral["Id"].to_numpy())
    previsoes_baseline = {
        "baseline_atual_21_19": juiz["pred_ancorada_publico"].to_numpy(),
        "baseline_arvores_corr30": geral["blend_arvores_alpha150_corr30"].to_numpy(),
        "baseline_xgb_alpha150": geral["xgboost_alpha150_bruto"].to_numpy(),
        "baseline_lgb_alpha150": geral["lightgbm_alpha150_bruto"].to_numpy(),
    }

    resumos, curvas = [], []
    oof = df[["Id", "preco"]].copy()
    oof["fold"] = folds
    for nome, pred in {**previsoes_baseline, **previsoes}.items():
        familia = "baseline" if nome.startswith("baseline") else nome.split("_")[0]
        linha, curva, calibrada = avaliar_previsao(nome, familia, y, pred, folds)
        chave_duracao = next((k for k in duracoes if k in nome), None)
        linha["duracao_grupo_segundos"] = duracoes.get(chave_duracao, 0.0)
        resumos.append(linha)
        curvas.extend(curva)
        oof[f"{nome}_bruto"] = pred
        oof[f"{nome}_calibrado_cf"] = calibrada

    pasta = os.path.join(raiz, "resultados")
    resumo = pd.DataFrame(resumos).sort_values("rmspe_950_calibrado_cf")
    resumo.to_csv(os.path.join(pasta, "especialista_caros_rodada1_resumo.csv"), index=False)
    pd.DataFrame(curvas).to_csv(
        os.path.join(pasta, "especialista_caros_rodada1_curvas.csv"), index=False
    )
    oof.to_csv(os.path.join(pasta, "especialista_caros_rodada1_oof.csv"), index=False)
    with open(
        os.path.join(pasta, "especialista_caros_rodada1_configuracoes.json"),
        "w", encoding="utf-8"
    ) as arquivo:
        json.dump({
            "arvores": CONFIGURACOES_ARVORES_RODADA1,
            "catboost": CONFIGURACOES_CAT_RODADA1,
            "xgb_device": "cpu",
        }, arquivo, ensure_ascii=False, indent=2)
    print("\nMelhores da rodada 1:")
    print(resumo.head(20).to_string(index=False))


def executar_rodada2():
    raiz, df = carregar_treino()
    splits = list(KFold(n_splits=5, shuffle=True, random_state=42).split(df))
    folds = np.zeros(len(df), dtype=int)
    for numero, (_, validacao) in enumerate(splits, start=1):
        folds[validacao] = numero
    y = df["preco"].to_numpy(dtype=float)
    previsoes = {}
    duracoes = {}
    for item in CONFIGURACOES_RODADA2:
        nome = item["nome"]
        modelo_nome = item["modelo"]
        config = {
            **CONFIGURACAO_RAW_950,
            "limite_treino": item.get("limite_treino", 950_000),
        }
        pred = np.full(len(df), np.nan)
        inicio = time.perf_counter()
        for numero, (it, iv) in enumerate(splits, start=1):
            pred[iv] = treinar_modelo_arvore_fold(
                df.iloc[it],
                df.iloc[iv],
                config,
                modelo_nome,
                overrides=item.get("overrides"),
                features_caras=item.get("features_caras", False),
                target_encoding=item.get("target_encoding", False),
            )
            print(f"{nome}: fold {numero}/5 concluido")
        duracoes[nome] = time.perf_counter() - inicio
        previsoes[nome] = pred

    rodada1 = pd.read_csv(
        os.path.join(raiz, "resultados", "especialista_caros_rodada1_oof.csv")
    )
    referencias = {
        "ref_xgb_raw_sub950": rodada1["xgb_raw_sub950_rmspe_bruto"].to_numpy(),
        "ref_lgb_raw_sub950": rodada1["lgb_raw_sub950_rmspe_bruto"].to_numpy(),
        "ref_arvores_corr30": rodada1["baseline_arvores_corr30_bruto"].to_numpy(),
    }
    resumos, curvas = [], []
    oof = df[["Id", "preco"]].copy()
    oof["fold"] = folds
    for nome, pred in {**referencias, **previsoes}.items():
        familia = "referencia" if nome.startswith("ref_") else nome.split("_")[0]
        linha, curva, calibrada = avaliar_previsao(nome, familia, y, pred, folds)
        linha["duracao_segundos"] = duracoes.get(nome, 0.0)
        resumos.append(linha)
        curvas.extend(curva)
        oof[f"{nome}_bruto"] = pred
        oof[f"{nome}_calibrado_cf"] = calibrada

    pasta = os.path.join(raiz, "resultados")
    resumo = pd.DataFrame(resumos).sort_values("rmspe_950_calibrado_cf")
    resumo.to_csv(os.path.join(pasta, "especialista_caros_rodada2_resumo.csv"), index=False)
    pd.DataFrame(curvas).to_csv(
        os.path.join(pasta, "especialista_caros_rodada2_curvas.csv"), index=False
    )
    oof.to_csv(os.path.join(pasta, "especialista_caros_rodada2_oof.csv"), index=False)
    with open(
        os.path.join(pasta, "especialista_caros_rodada2_configuracoes.json"),
        "w", encoding="utf-8"
    ) as arquivo:
        json.dump(CONFIGURACOES_RODADA2, arquivo, ensure_ascii=False, indent=2)
    print("\nMelhores da rodada 2:")
    print(resumo.head(20).to_string(index=False))


def executar_rodada2b():
    raiz, df = carregar_treino()
    splits = list(KFold(n_splits=5, shuffle=True, random_state=42).split(df))
    folds = np.zeros(len(df), dtype=int)
    for numero, (_, validacao) in enumerate(splits, start=1):
        folds[validacao] = numero
    y = df["preco"].to_numpy(dtype=float)
    previsoes, duracoes = {}, {}
    for item in CONFIGURACOES_RODADA2B:
        nome = item["nome"]
        config = {
            **CONFIGURACAO_RAW_950,
            "limite_treino": item.get("limite_treino", 900_000),
        }
        pred = np.full(len(df), np.nan)
        inicio = time.perf_counter()
        for numero, (it, iv) in enumerate(splits, start=1):
            pred[iv] = treinar_modelo_arvore_fold(
                df.iloc[it], df.iloc[iv], config, item["modelo"],
                overrides=item.get("overrides"),
                features_caras=item.get("features_caras", False),
                target_encoding=item.get("target_encoding", False),
            )
            print(f"{nome}: fold {numero}/5 concluido")
        duracoes[nome] = time.perf_counter() - inicio
        previsoes[nome] = pred

    rodada2 = pd.read_csv(
        os.path.join(raiz, "resultados", "especialista_caros_rodada2_oof.csv")
    )
    referencias = {
        "ref_xgb_thr900": rodada2["xgb_thr900_bruto"].to_numpy(),
        "ref_xgb_te950": rodada2["xgb_target_encoding_bruto"].to_numpy(),
        "ref_xgb_feat_te950": rodada2["xgb_features_caras_te_bruto"].to_numpy(),
        "ref_lgb31_950": rodada2["lgb_l31_d7_mc40_i600_bruto"].to_numpy(),
    }
    resumos, curvas = [], []
    oof = df[["Id", "preco"]].copy()
    oof["fold"] = folds
    for nome, pred in {**referencias, **previsoes}.items():
        familia = "referencia" if nome.startswith("ref_") else nome.split("_")[0]
        linha, curva, calibrada = avaliar_previsao(nome, familia, y, pred, folds)
        linha["duracao_segundos"] = duracoes.get(nome, 0.0)
        resumos.append(linha)
        curvas.extend(curva)
        oof[f"{nome}_bruto"] = pred
        oof[f"{nome}_calibrado_cf"] = calibrada
    pasta = os.path.join(raiz, "resultados")
    resumo = pd.DataFrame(resumos).sort_values("rmspe_950_calibrado_cf")
    resumo.to_csv(os.path.join(pasta, "especialista_caros_rodada2b_resumo.csv"), index=False)
    pd.DataFrame(curvas).to_csv(
        os.path.join(pasta, "especialista_caros_rodada2b_curvas.csv"), index=False
    )
    oof.to_csv(os.path.join(pasta, "especialista_caros_rodada2b_oof.csv"), index=False)
    with open(
        os.path.join(pasta, "especialista_caros_rodada2b_configuracoes.json"),
        "w", encoding="utf-8"
    ) as arquivo:
        json.dump(CONFIGURACOES_RODADA2B, arquivo, ensure_ascii=False, indent=2)
    print("\nMelhores da rodada 2b:")
    print(resumo.head(20).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rodada", choices=["rodada1", "rodada2", "rodada2b"])
    args = parser.parse_args()
    if args.rodada == "rodada1":
        executar_rodada1()
    elif args.rodada == "rodada2":
        executar_rodada2()
    elif args.rodada == "rodada2b":
        executar_rodada2b()


if __name__ == "__main__":
    main()
