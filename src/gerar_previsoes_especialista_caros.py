"""Treina o especialista caro congelado e preserva previsoes para o teste.

Este script nao cria submission e nao altera nenhum juiz. A calibracao final
e aprendida nas previsoes OOF da semente 42, a mesma semente dos modelos que
sao treinados com toda a base.
"""

import json
import os
import time

import numpy as np
import pandas as pd

from buscar_especialista_caros import (
    CONFIGURACAO_RAW_950,
    LIMITE_PRINCIPAL,
    carregar_treino,
    treinar_modelo_arvore_fold,
)
from refinar_especialista_caros import (
    aplicar_calibracao,
    aplicar_correspondencia,
    fit_calibracao,
)
from testar_catboost_alphas import corrigir_dados
from testar_catboost_correspondencias import prever_correspondencias


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


def carregar_teste(raiz):
    caminho = os.path.join(raiz, "data", "conjunto_de_teste (3).csv")
    return corrigir_dados(pd.read_csv(caminho), corrigir_alvo=False).reset_index(
        drop=True
    )


def aprender_calibracao_final(raiz, treino):
    caminho = os.path.join(
        raiz, "resultados", "especialista_caros_estabilidade_oof.csv"
    )
    oof = pd.read_csv(caminho)
    assert np.array_equal(treino["Id"].to_numpy(), oof["Id"].to_numpy())
    y = treino["preco"].to_numpy(dtype=float)
    bruto = oof["especialista_s42_bruto"].to_numpy(dtype=float)
    mascara = y >= LIMITE_PRINCIPAL
    return fit_calibracao(y[mascara], bruto[mascara], "log_afim")


def resumo_numerico(valores):
    valores = np.asarray(valores, dtype=float)
    return {
        "minimo": float(valores.min()),
        "p05": float(np.quantile(valores, 0.05)),
        "mediana": float(np.median(valores)),
        "p95": float(np.quantile(valores, 0.95)),
        "maximo": float(valores.max()),
        "media": float(valores.mean()),
    }


def executar():
    raiz, treino = carregar_treino()
    teste = carregar_teste(raiz)
    pasta = os.path.join(raiz, "resultados")
    os.makedirs(pasta, exist_ok=True)

    parametros_calibracao = aprender_calibracao_final(raiz, treino)
    inicio = time.perf_counter()
    pred_xgb = treinar_modelo_arvore_fold(
        treino, teste, CONFIG_XGB, "xgb"
    )
    pred_lgb = treinar_modelo_arvore_fold(
        treino, teste, CONFIG_LGB, "lgb", overrides=LGB_OVERRIDES
    )
    correspondencia = prever_correspondencias(treino, teste)
    xgb_corr = aplicar_correspondencia(pred_xgb, correspondencia, CORR_XGB)
    lgb_corr = aplicar_correspondencia(pred_lgb, correspondencia, CORR_LGB)
    blend_bruto = PESO_XGB * xgb_corr + (1.0 - PESO_XGB) * lgb_corr
    especialista = aplicar_calibracao(
        blend_bruto, "log_afim", parametros_calibracao
    )
    duracao = time.perf_counter() - inicio

    assert len(teste) == 2_000
    assert teste["Id"].is_unique
    for valores in [pred_xgb, pred_lgb, blend_bruto, especialista]:
        assert np.isfinite(valores).all()
        assert (valores > 0).all()

    saida = teste[["Id"]].copy()
    saida["xgb_caro_bruto"] = pred_xgb
    saida["lightgbm_caro_bruto"] = pred_lgb
    saida["preco_correspondencia"] = correspondencia
    saida["tem_correspondencia"] = np.isfinite(correspondencia).astype(int)
    saida["xgb_caro_corr50"] = xgb_corr
    saida["lightgbm_caro_corr50"] = lgb_corr
    saida["especialista_caro_blend_bruto"] = blend_bruto
    saida["especialista_caro"] = especialista
    caminho_saida = os.path.join(
        pasta, "especialista_caros_previsoes_teste.csv"
    )
    saida.to_csv(caminho_saida, index=False)

    configuracao = {
        "status": "congelado_validado",
        "gera_submission": False,
        "quantidade_treino_total": int(len(treino)),
        "quantidade_treino_xgb": int((treino["preco"] >= 900_000).sum()),
        "quantidade_treino_lgb": int((treino["preco"] >= 950_000).sum()),
        "quantidade_teste": int(len(teste)),
        "xgb": {
            "limite_treino": 900_000,
            "target": "preco/1e6",
            "sample_weight": "1/preco**2 normalizado",
            "peso_correspondencia": CORR_XGB,
        },
        "lightgbm": {
            "limite_treino": 950_000,
            "target": "preco/1e6",
            "sample_weight": "1/preco**2 normalizado",
            "peso_correspondencia": CORR_LGB,
            "overrides": LGB_OVERRIDES,
        },
        "blend": {"peso_xgb": PESO_XGB, "peso_lightgbm": 1.0 - PESO_XGB},
        "calibracao_final": {
            "tipo": "log_afim",
            "origem": "OOF semente 42, apenas preco >= 950000",
            **parametros_calibracao,
        },
        "correspondencia_teste": {
            "quantidade": int(np.isfinite(correspondencia).sum()),
            "proporcao": float(np.isfinite(correspondencia).mean()),
        },
        "previsoes_teste": resumo_numerico(especialista),
        "duracao_segundos": duracao,
        "arquivo_previsoes": os.path.basename(caminho_saida),
    }
    caminho_config = os.path.join(
        pasta, "especialista_caros_configuracao_final.json"
    )
    with open(caminho_config, "w", encoding="utf-8") as arquivo:
        json.dump(configuracao, arquivo, ensure_ascii=False, indent=2)

    print(json.dumps(configuracao, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    executar()
