
import os 
import numpy as np 
import pandas as pd
from lightgbm import LGBMRegressor
from pre_processamento import criar_features_modelo, x ,y

raiz = os.path.join(os.path.dirname(__file__), "..")

caminho_teste = os.path.join(
    raiz, "data", "conjunto_de_teste (3).csv"
)

caminho_exemplo = os.path.join(
    raiz, "data", "exemplo_arquivo_respostas.csv"
)

df_teste = pd.read_csv(caminho_teste)
ids_teste = df_teste["Id"].copy()

df_teste_modelo = criar_features_modelo(df_teste)

x_teste = df_teste_modelo.drop(columns = ["Id"])

x_teste = x_teste.reindex(
    columns=x.columns,
    fill_value=0
)

modelo_lgbm_final = LGBMRegressor(
    objective = "regression",
    n_estimators = 800,
    learning_rate = 0.03,
    num_leaves = 15,
    max_depth = 5,
    min_child_samples = 20,
    colsample_bytree = 0.8,
    reg_lambda = 1.0,
    random_state = 42,
    n_jobs = -1,
    verbosity = -1,
)

y_log = np.log1p(y)

modelo_lgbm_final.fit(x, y_log)

previsoes_log = modelo_lgbm_final.predict(x_teste)
previsoes_finais = np.expm1(previsoes_log)

submission = pd.read_csv(caminho_exemplo)

assert submission["Id"].equals(ids_teste)

submission["preco"] = previsoes_finais.round(2)

pasta_saida = os.path.join(raiz, "submissions")
os.makedirs(pasta_saida , exist_ok=True)

caminho_saida = os.path.join(
    pasta_saida, 
    "submission_lightgbm_800_arvores.csv"
)

submission.to_csv(caminho_saida, index = False)


                                                                                                                       
