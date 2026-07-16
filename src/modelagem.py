import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from pre_processamento import x, y


def calcular_rmspe(y_real, y_previsto):
    erros_percentuais = (y_real - y_previsto) / y_real
    erros_quadrados = erros_percentuais**2
    return np.sqrt(np.mean(erros_quadrados))


# A mesma divisao e usada nos dois experimentos.
x_treino, x_validacao, y_treino, y_validacao = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
)

# Regressao linear com todas as features e preco na escala original.
modelo_normal = LinearRegression()
modelo_normal.fit(x_treino, y_treino)

previsoes_normal = modelo_normal.predict(x_validacao)
rmspe_normal = calcular_rmspe(y_validacao, previsoes_normal)

print(f"RMSPE normal: {rmspe_normal:.4f}")
print(f"RMSPE percentual normal: {rmspe_normal * 100:.2f}%")
print(f"Previsoes negativas no modelo normal: {np.sum(previsoes_normal < 0)}")

# Regressao linear com todas as features e log do preco.
y_treino_log = np.log1p(y_treino)

modelo_log = LinearRegression()
modelo_log.fit(x_treino, y_treino_log)

previsoes_em_log = modelo_log.predict(x_validacao)
previsoes_log = np.expm1(previsoes_em_log)
rmspe_log = calcular_rmspe(y_validacao, previsoes_log)


print(f"RMSPE com log: {rmspe_log:.4f}")
print(f"RMSPE percentual com log: {rmspe_log * 100:.2f}%")
print(f"Previsoes negativas no modelo com log: {np.sum(previsoes_log < 0)}")

#Previsão utilizando XGboost
modelo_xgb_log = XGBRegressor(
    n_estimators = 300,
    learning_rate = 0.05,
    max_depth = 4,
    random_state = 42,
    n_jobs = -1

)

modelo_xgb_log.fit(x_treino, y_treino_log)

previsoes_xgb_em_log = modelo_xgb_log.predict(x_validacao)
previsoes_xgb = np.expm1(previsoes_xgb_em_log)

rmspe_xgb = calcular_rmspe(y_validacao, previsoes_xgb)

print(f"RMSPE XGBoost com log: {rmspe_xgb:.4f}")                                                                                                                       
print(f"RMSPE percentual: {rmspe_xgb * 100:.2f}%")

#Previsão utilizando LightGBM

modelo_lgbm_log = LGBMRegressor(
    objective = "regression",
    n_estimators = 300,
    learning_rate = 0.05,
    num_leaves = 15,
    max_depth = 5,
    min_child_samples = 20,
    colsample_bytree = 0.8,
    reg_lambda = 1.0,
    random_state = 42,
    n_jobs = -1,
    verbosity = -1,
)

modelo_lgbm_log.fit(x_treino,y_treino_log)

previsoes_lgbm_em_log = modelo_lgbm_log.predict(x_validacao)
previsoes_lgbm = np.expm1(previsoes_lgbm_em_log)

rmspe_lgbm = calcular_rmspe(y_validacao, previsoes_lgbm)

print(f"RMSPE LightGBM com log: {rmspe_lgbm:.4f}")
print(f"RMSPE percentual com log: {rmspe_lgbm * 100:.2f}%")

# LightGBM ajustado por busca aleatoria com validacao cruzada de 5 folds.
modelo_lgbm_ajustado = LGBMRegressor(
    objective="regression",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=7,
    min_child_samples=20,
    subsample=0.75,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.3,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)

modelo_lgbm_ajustado.fit(x_treino, y_treino_log)

previsoes_lgbm_ajustado_log = modelo_lgbm_ajustado.predict(x_validacao)
previsoes_lgbm_ajustado = np.expm1(previsoes_lgbm_ajustado_log)
rmspe_lgbm_ajustado = calcular_rmspe(y_validacao, previsoes_lgbm_ajustado)

print(f"RMSPE LightGBM ajustado: {rmspe_lgbm_ajustado:.4f}")
print(f"RMSPE percentual ajustado: {rmspe_lgbm_ajustado * 100:.2f}%")

# Teste direcionado: mais arvores com uma taxa de aprendizado menor.
modelo_lgbm_800_arvores = LGBMRegressor(
    objective="regression",
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=15,
    max_depth=5,
    min_child_samples=20,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)

modelo_lgbm_800_arvores.fit(x_treino, y_treino_log)

previsoes_lgbm_800_log = modelo_lgbm_800_arvores.predict(x_validacao)
previsoes_lgbm_800 = np.expm1(previsoes_lgbm_800_log)
rmspe_lgbm_800 = calcular_rmspe(y_validacao, previsoes_lgbm_800)

print(f"RMSPE LightGBM com 800 arvores: {rmspe_lgbm_800:.4f}")
print(f"RMSPE percentual com 800 arvores: {rmspe_lgbm_800 * 100:.2f}%")


















































