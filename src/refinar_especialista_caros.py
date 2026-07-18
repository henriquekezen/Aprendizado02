"""Refina calibracao, correspondencia e blends do especialista caro."""

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from buscar_especialista_caros import (
    LIMITE_CAUDA,
    LIMITE_PRINCIPAL,
    LIMITES_CURVA,
    calcular_fator,
    carregar_treino,
    rmspe,
)


PESOS_CORRESPONDENCIA = [0.0, 0.15, 0.30]


def aplicar_correspondencia(pred, correspondencia, peso):
    saida = np.asarray(pred, dtype=float).copy()
    mascara = np.isfinite(correspondencia)
    saida[mascara] = (
        (1.0 - peso) * saida[mascara] + peso * correspondencia[mascara]
    )
    return saida


def fit_log_afim(y, p):
    y = np.asarray(y, dtype=float)
    p = np.maximum(np.asarray(p, dtype=float), 1.0)
    logp = np.log1p(p)
    centro = float(logp.mean())

    def objetivo(parametros):
        a, b = parametros
        ajustada = np.expm1(logp + a + b * (logp - centro))
        return rmspe(y, ajustada) ** 2

    resultado = minimize(
        objetivo,
        x0=np.array([0.0, 0.0]),
        method="L-BFGS-B",
        bounds=[(-0.40, 0.40), (-0.20, 0.20)],
    )
    assert resultado.success
    return {
        "a": float(resultado.x[0]),
        "b": float(resultado.x[1]),
        "centro": centro,
    }


def aplicar_log_afim(p, parametros):
    logp = np.log1p(np.maximum(np.asarray(p, dtype=float), 1.0))
    return np.expm1(
        logp
        + parametros["a"]
        + parametros["b"] * (logp - parametros["centro"])
    )


def fit_calibracao(y, p, tipo):
    if tipo == "multiplicativa":
        return {"fator": calcular_fator(y, p)}
    if tipo == "log_afim":
        return fit_log_afim(y, p)
    raise ValueError(tipo)


def aplicar_calibracao(p, tipo, parametros):
    if tipo == "multiplicativa":
        return np.asarray(p, dtype=float) * parametros["fator"]
    if tipo == "log_afim":
        return aplicar_log_afim(p, parametros)
    raise ValueError(tipo)


def calibrar_crossfit(y, p, folds, tipo):
    saida = np.full(len(y), np.nan)
    parametros_folds = []
    for fold in sorted(np.unique(folds)):
        treino = (folds != fold) & (y >= LIMITE_PRINCIPAL)
        validacao = folds == fold
        parametros = fit_calibracao(y[treino], p[treino], tipo)
        saida[validacao] = aplicar_calibracao(p[validacao], tipo, parametros)
        parametros_folds.append({"fold": int(fold), **parametros})
    final = fit_calibracao(y[y >= LIMITE_PRINCIPAL], p[y >= LIMITE_PRINCIPAL], tipo)
    return saida, parametros_folds, final


def resumo_calibrado(nome, familia, y, pred, folds, extras=None):
    principal = y >= LIMITE_PRINCIPAL
    cauda = y >= LIMITE_CAUDA
    linha = {
        "nome": nome,
        "familia": familia,
        "rmspe_950": rmspe(y[principal], pred[principal]),
        "rmspe_1300": rmspe(y[cauda], pred[cauda]),
        **(extras or {}),
    }
    for fold in sorted(np.unique(folds)):
        mascara = (folds == fold) & principal
        linha[f"rmspe_fold_{int(fold)}"] = rmspe(y[mascara], pred[mascara])
    curvas = []
    for limite in LIMITES_CURVA:
        mascara = y >= limite
        curvas.append({
            "nome": nome,
            "familia": familia,
            "limite_preco": limite,
            "quantidade": int(mascara.sum()),
            "rmspe": rmspe(y[mascara], pred[mascara]),
        })
    return linha, curvas


def blend_crossfit(y, p1, p2, folds, tipo_calibracao):
    saida = np.full(len(y), np.nan)
    parametros_folds = []
    grade = np.linspace(0.0, 1.0, 21)
    for fold in sorted(np.unique(folds)):
        treino = (folds != fold) & (y >= LIMITE_PRINCIPAL)
        validacao = folds == fold
        melhor = None
        for peso1 in grade:
            bruto = peso1 * p1 + (1.0 - peso1) * p2
            parametros = fit_calibracao(
                y[treino], bruto[treino], tipo_calibracao
            )
            ajustada = aplicar_calibracao(
                bruto[treino], tipo_calibracao, parametros
            )
            erro = rmspe(y[treino], ajustada)
            if melhor is None or erro < melhor["rmspe_treino"]:
                melhor = {
                    "peso1": float(peso1),
                    "rmspe_treino": float(erro),
                    "parametros": parametros,
                }
        bruto_validacao = (
            melhor["peso1"] * p1[validacao]
            + (1.0 - melhor["peso1"]) * p2[validacao]
        )
        saida[validacao] = aplicar_calibracao(
            bruto_validacao, tipo_calibracao, melhor["parametros"]
        )
        parametros_folds.append({
            "fold": int(fold),
            "peso1": melhor["peso1"],
            "rmspe_treino": melhor["rmspe_treino"],
            **melhor["parametros"],
        })

    mascara = y >= LIMITE_PRINCIPAL
    melhor_final = None
    for peso1 in grade:
        bruto = peso1 * p1 + (1.0 - peso1) * p2
        parametros = fit_calibracao(y[mascara], bruto[mascara], tipo_calibracao)
        ajustada = aplicar_calibracao(
            bruto[mascara], tipo_calibracao, parametros
        )
        erro = rmspe(y[mascara], ajustada)
        if melhor_final is None or erro < melhor_final["rmspe"]:
            melhor_final = {
                "peso1": float(peso1),
                "rmspe": float(erro),
                **parametros,
            }
    return saida, parametros_folds, melhor_final


def executar():
    raiz, df = carregar_treino()
    pasta = os.path.join(raiz, "resultados")
    r2 = pd.read_csv(os.path.join(pasta, "especialista_caros_rodada2_oof.csv"))
    correspondencias_df = pd.read_csv(
        os.path.join(pasta, "catboost_correspondencias_oof.csv")
    )
    assert np.array_equal(df["Id"].to_numpy(), r2["Id"].to_numpy())
    y = df["preco"].to_numpy(dtype=float)
    folds = r2["fold"].to_numpy(dtype=int)
    correspondencia = correspondencias_df["previsao_correspondencia"].to_numpy()
    candidatos = {
        "xgb900": r2["xgb_thr900_bruto"].to_numpy(),
        "xgb_te950": r2["xgb_target_encoding_bruto"].to_numpy(),
        "xgb_feat_te950": r2["xgb_features_caras_te_bruto"].to_numpy(),
        "lgb31_950": r2["lgb_l31_d7_mc40_i600_bruto"].to_numpy(),
    }
    resumos, curvas, parametros_todos = [], [], {}
    oof = df[["Id", "preco"]].copy()
    oof["fold"] = folds
    melhores_corr = {}

    for nome_base, pred_base in candidatos.items():
        resultados_base = []
        for peso_corr in PESOS_CORRESPONDENCIA:
            pred_corr = aplicar_correspondencia(
                pred_base, correspondencia, peso_corr
            )
            for tipo in ["multiplicativa", "log_afim"]:
                pred_cf, params_folds, params_final = calibrar_crossfit(
                    y, pred_corr, folds, tipo
                )
                nome = f"{nome_base}_corr{int(peso_corr*100):02d}_{tipo}"
                linha, curva = resumo_calibrado(
                    nome, "individual", y, pred_cf, folds,
                    {"peso_correspondencia": peso_corr,
                     "calibracao": tipo}
                )
                resumos.append(linha)
                curvas.extend(curva)
                resultados_base.append((linha["rmspe_950"], peso_corr, tipo, pred_corr))
                parametros_todos[nome] = {
                    "folds": params_folds,
                    "final": params_final,
                }
                oof[f"{nome}_calibrado_cf"] = pred_cf
        melhor = min(resultados_base, key=lambda item: item[0])
        melhores_corr[nome_base] = {
            "peso_correspondencia": melhor[1],
            "calibracao_individual": melhor[2],
            "pred_bruta_corr": melhor[3],
        }

    pares = [
        ("xgb900", "xgb_te950"),
        ("xgb900", "xgb_feat_te950"),
        ("xgb900", "lgb31_950"),
        ("xgb_feat_te950", "lgb31_950"),
    ]
    for nome1, nome2 in pares:
        p1 = melhores_corr[nome1]["pred_bruta_corr"]
        p2 = melhores_corr[nome2]["pred_bruta_corr"]
        for tipo in ["multiplicativa", "log_afim"]:
            pred_cf, params_folds, params_final = blend_crossfit(
                y, p1, p2, folds, tipo
            )
            nome = f"blend_{nome1}__{nome2}_{tipo}"
            linha, curva = resumo_calibrado(
                nome, "blend", y, pred_cf, folds,
                {
                    "peso_correspondencia_1": melhores_corr[nome1]["peso_correspondencia"],
                    "peso_correspondencia_2": melhores_corr[nome2]["peso_correspondencia"],
                    "calibracao": tipo,
                    "peso1_final": params_final["peso1"],
                },
            )
            resumos.append(linha)
            curvas.extend(curva)
            parametros_todos[nome] = {
                "componente1": nome1,
                "componente2": nome2,
                "folds": params_folds,
                "final": params_final,
            }
            oof[f"{nome}_calibrado_cf"] = pred_cf

    resumo = pd.DataFrame(resumos).sort_values("rmspe_950")
    resumo.to_csv(os.path.join(pasta, "especialista_caros_finalistas_resumo.csv"), index=False)
    pd.DataFrame(curvas).to_csv(
        os.path.join(pasta, "especialista_caros_finalistas_curvas.csv"), index=False
    )
    oof.to_csv(os.path.join(pasta, "especialista_caros_finalistas_oof.csv"), index=False)
    with open(
        os.path.join(pasta, "especialista_caros_finalistas_parametros.json"),
        "w", encoding="utf-8"
    ) as arquivo:
        json.dump(parametros_todos, arquivo, ensure_ascii=False, indent=2)
    with open(
        os.path.join(pasta, "especialista_caros_melhores_correspondencias.json"),
        "w", encoding="utf-8"
    ) as arquivo:
        json.dump(
            {nome: {k: v for k, v in dados.items() if k != "pred_bruta_corr"}
             for nome, dados in melhores_corr.items()},
            arquivo, ensure_ascii=False, indent=2
        )
    print(resumo.head(20).to_string(index=False))


if __name__ == "__main__":
    executar()
