# Modelagem

## Objetivo desta etapa

Depois da limpeza e da preparação dos dados, iniciamos a modelagem com duas regressões lineares e, em seguida, testamos XGBoost e LightGBM. Todos os modelos utilizam as 91 features disponíveis e exatamente os mesmos imóveis de treino e validação.

- modelo normal: aprende diretamente o preço em reais;
- modelo com log: aprende `log1p(preco)` e depois converte a previsão novamente para reais.

A comparação entre as duas regressões permite medir isoladamente o efeito da transformação logarítmica do alvo. Como o log apresentou o melhor resultado linear, ele também foi utilizado no treinamento dos dois modelos de boosting.

## Dados utilizados

O arquivo `modelagem.py` importa `x` e `y` de `pre_processamento.py`:

- `x` contém 91 características numéricas, incluindo as features criadas e as colunas one-hot;
- `y` contém o preço verdadeiro dos imóveis.

Os 4.654 imóveis foram divididos com `train_test_split`, usando `test_size=0.2` e `random_state=42`. O resultado foi:

| Conjunto | Imóveis | Features |
| :--- | ---: | ---: |
| Treino | 3.723 | 91 |
| Validação | 931 | 91 |

O `random_state` fixo garante que todos os modelos sejam comparados sobre as mesmas linhas.

## Métrica RMSPE

Os modelos são avaliados com o RMSPE, sigla para *Root Mean Squared Percentage Error*. Para cada imóvel, calculamos o erro proporcional ao preço verdadeiro:

```text
erro_percentual = (preco_real - preco_previsto) / preco_real
```

Depois, os erros são elevados ao quadrado, calculamos a média e aplicamos a raiz quadrada:

```text
RMSPE = raiz(media(erros_percentuais²))
```

Como os erros são elevados ao quadrado, previsões proporcionalmente muito ruins têm maior impacto no resultado.

## Regressão linear na escala normal

O primeiro modelo utiliza `LinearRegression` e aprende diretamente os preços em reais. Ele obteve:

```text
RMSPE: 0,4021
RMSPE percentual: 40,21%
Previsões negativas: 10
```

O resultado serve como referência para o uso das 91 features na escala original do preço. As dez previsões negativas mostram uma limitação importante: uma regressão linear sem restrições pode produzir valores matematicamente válidos, mas impossíveis no contexto de preços de imóveis.

## Regressão linear com log do preço

No segundo experimento, aplicamos `np.log1p` somente aos preços de treino. O modelo aprende a prever nessa escala e suas saídas são convertidas de volta para reais com `np.expm1` antes do cálculo do RMSPE:

```text
preco de treino -> log1p -> treinamento
previsão em log -> expm1 -> previsão em reais -> RMSPE
```

Esse modelo obteve:

```text
RMSPE: 0,3013
RMSPE percentual: 30,13%
Previsões negativas: 0
```

## XGBoost com log do preço

O terceiro experimento utiliza `XGBRegressor`. O XGBoost constrói árvores sequencialmente: cada nova árvore tenta corrigir parte dos erros deixados pelas anteriores. O modelo recebeu as 91 features sem escalonamento e foi treinado com `y_treino_log`.

Os parâmetros iniciais utilizados foram:

```text
n_estimators = 300
learning_rate = 0,05
max_depth = 4
random_state = 42
n_jobs = -1
```

As 300 árvores aplicam correções pequenas, controladas pelo `learning_rate`. A profundidade máxima de 4 limita a complexidade de cada árvore. Esses valores formam uma configuração inicial e ainda não foram otimizados.

Depois da previsão, aplicamos `np.expm1` para retornar à escala de reais antes do cálculo do RMSPE. O resultado interno foi:

```text
RMSPE: 0,2526
RMSPE percentual: 25,26%
Previsões negativas: 0
```

## LightGBM com log do preço

O quarto experimento utiliza `LGBMRegressor`. Assim como o XGBoost, o LightGBM é um algoritmo de boosting e constrói árvores que corrigem os erros acumulados. Uma diferença importante é sua estratégia de crescimento das árvores, que prioriza a folha com maior possibilidade de reduzir o erro. Por isso, `num_leaves` é um parâmetro central.

Os parâmetros iniciais utilizados foram:

```text
objective = regression
n_estimators = 300
learning_rate = 0,05
num_leaves = 15
max_depth = 5
min_child_samples = 20
colsample_bytree = 0,8
reg_lambda = 1,0
random_state = 42
n_jobs = -1
verbosity = -1
```

O limite de 15 folhas, a profundidade máxima de 5, a quantidade mínima de 20 imóveis por folha e a regularização L2 foram usados para reduzir o risco de sobreajuste. Cada árvore teve acesso a 80% das features por meio de `colsample_bytree`. Esses parâmetros também são apenas uma configuração inicial.

O LightGBM foi treinado com `y_treino_log`, e suas previsões foram convertidas para reais com `np.expm1`. O resultado interno foi:

```text
RMSPE: 0,2505
RMSPE percentual: 25,05%
Previsões negativas: 0
```

## Comparação dos resultados internos

Os valores abaixo foram calculados sobre os 931 imóveis da validação interna. Eles ainda não representam resultados do Kaggle nem de validação cruzada.

| Modelo | Features | Escala do alvo | RMSPE | Previsões negativas |
| :--- | ---: | :--- | ---: | ---: |
| Regressão linear normal | 91 | Preço em reais | 40,21% | 10 |
| Regressão linear com log | 91 | `log1p(preco)` | 30,13% | 0 |
| XGBoost com log | 91 | `log1p(preco)` | 25,26% | 0 |
| LightGBM com log | 91 | `log1p(preco)` | 25,05% | 0 |
| LightGBM ajustado com log | 91 | `log1p(preco)` | **24,26%** | 0 |
| LightGBM com 800 árvores | 91 | `log1p(preco)` | 24,52% | 0 |

Com todas as features disponíveis, a transformação logarítmica reduziu o RMSPE em 10,08 pontos percentuais e eliminou as previsões negativas. Isso representa uma redução relativa de aproximadamente 25% no erro da referência normal.

O resultado também mostra por que uma transformação deve ser avaliada junto com as informações fornecidas ao modelo. Com bairro, tipo e demais características disponíveis, o modelo em log conseguiu representar melhor as diferenças proporcionais entre imóveis baratos e caros.

O XGBoost reduziu o RMSPE em 4,87 pontos percentuais em relação à regressão linear com log. O LightGBM inicial superou o XGBoost por apenas 0,21 ponto percentual, uma diferença pequena que poderia depender da divisão específica dos dados. Depois do ajuste, o LightGBM chegou a 24,26% no mesmo holdout e a comparação passou a contar também com validação cruzada.

## Ajuste de hiperparâmetros do LightGBM

Para reduzir a dependência da divisão única de treino e validação, foi criado o script `src/ajustar_lightgbm.py`. Ele utiliza validação cruzada com cinco folds embaralhados e `random_state=42`. Em cada rodada, quatro folds são usados para treinamento e um para validação, de modo que todos os imóveis participem uma vez da validação.

O alvo continua sendo treinado como `log1p(preco)`, mas o scorer converte tanto os valores reais quanto as previsões com `expm1` antes de calcular o RMSPE. Isso garante que a busca otimize a mesma métrica usada na avaliação final, na escala original dos preços.

Primeiro, a configuração original foi avaliada nos cinco folds:

```text
RMSPE médio: 24,34%
Desvio entre folds: 1,41 ponto percentual
```

Em seguida, `RandomizedSearchCV` avaliou 40 combinações, totalizando 200 treinamentos. Foram variados os parâmetros ligados à quantidade e à complexidade das árvores, taxa de aprendizado, amostragem de linhas e colunas e regularização. A melhor combinação encontrada foi:

```text
n_estimators = 300
learning_rate = 0,05
num_leaves = 31
max_depth = 7
min_child_samples = 20
subsample = 0,75
subsample_freq = 1
colsample_bytree = 0,8
reg_alpha = 0,3
reg_lambda = 0,1
```

Seu resultado foi:

```text
RMSPE médio: 23,78%
Desvio entre folds: 1,38 ponto percentual
RMSPE no holdout original: 24,26%
```

O ganho médio na validação cruzada foi de 0,56 ponto percentual, enquanto o ganho no holdout original foi de 0,79 ponto percentual. Os resultados das 40 combinações foram salvos em `resultados/busca_hiperparametros_lightgbm.csv`, ordenados pelo RMSPE médio de validação.

## Busca direcionada por mais árvores

Depois que o primeiro ajuste não melhorou o resultado público, foi feito um teste isolando a relação entre `learning_rate` e `n_estimators`. Aumentar apenas a quantidade de árvores com a mesma taxa pode prolongar o aprendizado além do ponto útil. Por isso, taxas menores foram combinadas com quantidades progressivamente maiores de árvores.

O script `src/ajustar_arvores_lightgbm.py` avaliou 20 combinações com validação cruzada de cinco folds. O intervalo cobriu taxas de aprendizado de 0,05 a 0,01 e quantidades de 300 a 4.000 árvores, mantendo as demais características da configuração inicial.

Alguns pontos da curva foram:

| `learning_rate` | Árvores | RMSPE médio |
| ---: | ---: | ---: |
| 0,05 | 300 | 24,34% |
| 0,05 | 500 | 24,23% |
| 0,05 | 800 | 24,32% |
| 0,03 | 500 | 24,20% |
| 0,03 | 800 | **24,08%** |
| 0,03 | 1.200 | 24,11% |
| 0,02 | 1.200 | 24,14% |
| 0,01 | 2.400 | 24,13% |
| 0,01 | 4.000 | 24,21% |

O melhor ponto foi `learning_rate=0,03` com 800 árvores. Depois dele, aumentar a quantidade de árvores voltou a elevar o erro. Isso confirma que mais árvores podem ajudar, mas existe um limite e a taxa de aprendizado precisa ser ajustada em conjunto.

Essa configuração obteve 24,08% na média dos folds, com desvio de 1,51 ponto percentual, e 24,52% no holdout original. Os resultados completos foram salvos em `resultados/busca_arvores_lightgbm.csv`.

Foi gerado o arquivo `submissions/submission_lightgbm_800_arvores.csv`. Ele contém 2.000 IDs únicos, nenhum valor ausente e nenhum preço negativo. Como essa configuração altera menos a estrutura do modelo, suas previsões possuem correlação de 0,9983 com a primeira submission. Seu resultado público ainda precisa ser medido.

## Bairro como categoria nativa do LightGBM

O one-hot encoding transformava os 66 bairros em 66 colunas binárias. Essa representação funciona, mas dificulta que uma árvore trate em conjunto bairros com comportamento semelhante e pode favorecer a memorização de bairros com poucos exemplos.

Foi adicionada em `pre_processamento.py` uma transformação alternativa que mantém `bairro` como uma única coluna do tipo categórico do pandas. O `tipo` do imóvel continua usando one-hot encoding e todas as seis features criadas anteriormente foram mantidas. Essa mudança reduziu a entrada de 91 para 26 features sem retirar a informação do bairro.

O LightGBM foi treinado com `categorical_feature=["bairro"]`. Dessa forma, os códigos internos das categorias não são interpretados como números ordenados. O algoritmo pode procurar separações entre grupos de bairros durante a construção das árvores.

Também foram avaliados diferentes limites para agrupar bairros raros em `Outros`. Em cada fold, a contagem foi calculada somente no respectivo conjunto de treino. Bairros raros ou ausentes daquele treino foram transformados em `Outros` na validação, evitando usar informações do fold de validação durante a preparação.

O script `src/comparar_bairro_categorico.py` produziu os seguintes resultados:

| Estratégia | RMSPE médio | Desvio | Features |
| :--- | ---: | ---: | ---: |
| One-hot atual | 24,08% | 1,51 p.p. | 91 |
| Categórico sem agrupamento | 23,52% | 1,48 p.p. | 26 |
| Categórico, mínimo 5 | 23,53% | 1,55 p.p. | 26 |
| Categórico, mínimo 10 | **23,39%** | **1,37 p.p.** | 26 |
| Categórico, mínimo 15 | 23,42% | 1,46 p.p. | 26 |
| Categórico, mínimo 20 | 23,41% | 1,52 p.p. | 26 |

O limite de 10 imóveis reduziu o RMSPE médio em 0,69 ponto percentual e também reduziu a variação entre os folds. Ele superou o one-hot em todos os cinco folds, com ganhos individuais entre 0,06 e 1,61 ponto percentual. Os resultados completos estão em `resultados/comparacao_bairro_categorico.csv`.

No treinamento com toda a base, 36 bairros foram mantidos e os demais foram agrupados. Isso colocou 111 imóveis de treino e 38 imóveis de teste em `Outros`, incluindo bairros do teste que não haviam aparecido no treino.

O script `src/gerar_submission_bairro_categorico.py` treinou o LightGBM de 800 árvores com essa representação e gerou `submissions/submission_lightgbm_bairro_categorico.csv`. O arquivo contém 2.000 IDs únicos, nenhum valor ausente e nenhum preço negativo. Seu resultado no Kaggle ainda precisa ser medido.

No Kaggle, essa submission melhorou 0,06 ponto percentual em relação ao modelo one-hot de 800 árvores. O ganho público foi menor que o ganho da validação cruzada, mas confirmou que a representação categórica não piorou a generalização no conjunto oculto.

## Otimização controlada do modelo categórico

Depois da mudança de representação, os parâmetros foram testados sequencialmente em `src/otimizar_lightgbm_categorico.py`. Em cada etapa, somente um parâmetro foi alterado e o restante permaneceu congelado. Uma alternativa só poderia ser aceita se reduzisse a média e vencesse a configuração anterior em pelo menos três dos cinco folds.

Os valores padrão já eram os melhores para os quatro parâmetros específicos de categorias:

```text
cat_smooth = 10
cat_l2 = 10
min_data_per_group = 100
max_cat_threshold = 32
```

Valores menores e maiores foram testados, mas nenhum melhorou a referência de maneira consistente. Isso evita adicionar complexidade sem evidência de ganho.

Mantendo `learning_rate=0,03`, a quantidade de árvores apresentou:

| Árvores | RMSPE médio |
| ---: | ---: |
| 400 | 23,35% |
| 600 | **23,34%** |
| 800 | 23,39% |
| 1.000 | 23,45% |
| 1.200 | 23,49% |
| 1.600 | 23,68% |

O modelo de 600 árvores também reduziu o desvio entre os folds de 1,37 para 1,28 ponto percentual e elevou o RMSPE de treino de 17,97% para 18,68%. A menor distância entre treino e validação indica menos sobreajuste. Essa configuração gerou `submissions/submission_lightgbm_bairro_categorico_600.csv`.

Também foram avaliadas duas formas de combinar modelos. A média de modelos categóricos treinados com sementes diferentes piorou a validação e foi descartada. Um blend logarítmico com 90% do modelo categórico de 600 árvores e 10% do modelo one-hot chegou a 23,32%, ganho de apenas 0,02 ponto percentual. Como o ganho é pequeno, ele foi mantido somente como segunda candidata em `submissions/submission_blend_90cat_10onehot.csv`.

Os resultados detalhados estão em:

- `resultados/otimizacao_sequencial_lightgbm_categorico.csv`;
- `resultados/comparacao_blend_lightgbm.csv`;
- `resultados/comparacao_ensemble_sementes.csv`.

## Experimento com todas as linhas

Como teste separado, os 29 imóveis removidos durante a auditoria foram recolocados sem correção. O restante da configuração foi mantido igual ao melhor modelo público até o momento: todas as features criadas, bairro categórico com mínimo de 10 imóveis, 800 árvores e `learning_rate=0,03`.

O experimento foi implementado em `src/gerar_submission_todos_dados.py` e lê diretamente o CSV original com 4.683 imóveis. Ele não modifica `pre_processamento.py`, não altera a base limpa de 4.654 linhas e não substitui nenhuma submission anterior.

Na validação cruzada, os resultados foram:

```text
Fold 1: 30,62%
Fold 2: 26,27%
Fold 3: 24,22%
Fold 4: 755,15%
Fold 5: 26,10%
Média: 172,47%
Desvio: 291,35 pontos percentuais
```

O quarto fold foi dominado por preços extremamente baixos e inconsistentes, como o imóvel registrado por R$ 750. Como o RMSPE divide o erro pelo preço real, uma previsão normal para esse caso produz um erro percentual enorme. Portanto, essa validação mostra que recolocar indiscriminadamente todas as linhas prejudica fortemente a métrica interna.

Mesmo assim, foi gerada a submission experimental `submissions/submission_lightgbm_categorico_800_todos_dados.csv` para medir diretamente o efeito no conjunto oculto. Ela possui 2.000 IDs únicos, nenhum valor ausente e nenhum preço negativo. Suas previsões têm correlação de 0,9931 com o modelo limpo e diferem dele, em média, cerca de R$ 27,9 mil.

## Experimento com quase todas as linhas

Como alternativa intermediária, foi criada uma lista mínima contendo somente seis erros considerados praticamente incontestáveis:

| ID | Motivo da exclusão |
| ---: | :--- |
| 5910 | Preço de R$ 750 |
| 2405 | Preço de R$ 630 milhões para 98 m² |
| 4568 | Preço de R$ 65 milhões para 36 m² |
| 6004 | Preço de R$ 340 milhões para 72 m² |
| 6654 | 17.450 m² de área extra |
| 6383 | 30 vagas |

As outras 23 linhas anteriormente removidas foram recuperadas sem alteração. O experimento utilizou 4.677 dos 4.683 imóveis originais e manteve o modelo categórico de 800 árvores.

A validação cruzada resultou em:

```text
Fold 1: 22,59%
Fold 2: 23,37%
Fold 3: 27,11%
Fold 4: 22,16%
Fold 5: 24,76%
Média: 24,00%
Desvio: 1,79 ponto percentual
```

O resultado é muito mais estável que o experimento com todas as linhas, mas ainda ficou acima dos 23,39% obtidos com a base limpa. Como os conjuntos avaliados não possuem exatamente as mesmas linhas, a comparação não é perfeitamente pareada, porém a maior variação ainda indica que parte dos casos recuperados pode adicionar ruído.

O script `src/gerar_submission_quase_todos_dados.py` gerou `submissions/submission_lightgbm_categorico_800_quase_todos_dados.csv`. O arquivo possui 2.000 IDs únicos, nenhum valor ausente e nenhum preço negativo. Suas previsões têm correlação de 0,9954 com o modelo limpo e diferem dele, em média, cerca de R$ 18,7 mil. O pipeline principal permanece inalterado.

No Kaggle, essa versão melhorou mais de 1,5 ponto percentual em relação ao melhor modelo anterior, apesar de sua validação cruzada aleatória ter sido pior. Isso motivou uma auditoria da estratégia de validação.

## Por que a validação divergiu do Kaggle

Os resultados de 23,39% da base limpa e 24,00% da base quase completa não eram diretamente comparáveis: cada modelo havia sido avaliado sobre um conjunto de linhas diferente. A base limpa não precisava prever os 23 casos recuperados, enquanto a base quase completa precisava.

O script `src/analisar_validacao_limpeza.py` refez a comparação usando exatamente os mesmos imóveis limpos nos cinco conjuntos de validação. As 23 linhas recuperadas foram adicionadas somente ao treino do segundo modelo:

```text
Treino limpo: 23,39%
Treino com 23 linhas recuperadas: 23,38%
Diferença: melhora de 0,02 ponto percentual
Folds vencidos pelo modelo com recuperados: 2 de 5
```

Portanto, adicionar as linhas não prejudicava o modelo nas linhas comuns. O resultado anterior piorava porque as 23 linhas recuperadas eram extremamente difíceis de prever. Um modelo treinado na base limpa obteve 192,47% de RMSPE quando avaliado somente nesses 23 casos. Os cinco casos com maior erro concentraram 44,3% de todo o erro quadrático desse grupo.

### Estrutura do split

Treino e teste formam uma sequência completa de 6.683 IDs, sem lacunas ou duplicatas, mas não foram separados aleatoriamente:

```text
Teste: aproximadamente IDs 0 a 1999
Treino: IDs 2000 a 6682
```

O `KFold(shuffle=True)` mistura apenas os IDs do bloco de treino e não reproduz essa direção. Para aproximar o Kaggle, foram criados holdouts com os menores IDs do treino como validação:

| Bloco de IDs baixos | Treino limpo | Treino com recuperados | Ganho |
| ---: | ---: | ---: | ---: |
| 10% | 23,85% | 23,00% | 0,85 p.p. |
| 15% | 22,64% | 21,98% | 0,67 p.p. |
| 20% | 25,09% | 22,31% | 2,78 p.p. |
| 25% | 24,48% | 22,61% | 1,87 p.p. |
| 30% | 24,42% | 22,42% | 2,00 p.p. |

Os cinco cortes favoreceram as linhas recuperadas. Esses ganhos estão muito mais próximos da melhora observada no Kaggle que a validação aleatória original.

### Erros sistemáticos repetidos

O teste contém exemplos praticamente idênticos aos casos que haviam sido removidos. O imóvel de teste `570` possui 569 m², dois quartos, uma suíte, uma vaga e fica em Iputinga. Ele é uma cópia exata das features dos imóveis de treino `3937` e `3984`, cujos preços são R$ 235 mil e R$ 240 mil.

```text
Previsão do modelo limpo para o ID 570: R$ 676,7 mil
Previsão com as linhas recuperadas: R$ 239,4 mil
```

No total, 433 dos 2.000 imóveis de teste possuem uma combinação completa de features que também aparece no treino, formando 843 pares exatos. Isso indica repetição de anúncios ou registros e torna exemplos aparentemente anômalos úteis para prever registros repetidos no conjunto oculto.

Outro exemplo é o ID de teste `1245`, com 577 m², dois quartos e bairro Sto Amaro. O treino recuperado contém o ID `4220` com o mesmo padrão principal e preço de R$ 371 mil. A previsão caiu de R$ 1,04 milhão no modelo limpo para R$ 289 mil no modelo quase completo.

As distribuições marginais de treino e teste são próximas, com estatística KS máxima de 0,042 entre as features numéricas analisadas. Assim, uma mudança geral de distribuição parece menos importante que a combinação de split por bloco, registros repetidos e erros sistemáticos nas caudas.

### Conclusão metodológica

A remoção por `preco_m2` utilizou o próprio alvo para selecionar linhas e depois mediu a validação somente na população restante. Isso estima o erro condicionado a um dado ser considerado limpo, enquanto o Kaggle avalia a população original, que ainda contém registros com os mesmos padrões problemáticos.

Para os próximos experimentos, a validação deve combinar:

- folds aleatórios sobre um conjunto de validação comum;
- holdout por blocos de ID, aproximando a direção treino-teste;
- análise separada dos casos extremos, sem deixá-los dominar a média principal;
- regras de limpeza aprendidas apenas no treino de cada fold quando dependerem do alvo.

Os resultados detalhados foram salvos em `resultados/comparacao_validacao_mesmas_linhas.csv`, `resultados/auditoria_23_linhas_recuperadas.csv` e `resultados/comparacao_holdout_blocos_id.csv`.

## Correção conservadora de erros de escala

A mediana de preço por m² do bairro foi utilizada para auditar se os valores extremos eram compatíveis com a hipótese de uma casa decimal perdida. Entre as 23 linhas recuperadas, 21 apresentavam preço por m² entre 6% e 14% da mediana do bairro. Depois de dividir `area_util` por 10, passaram para 64%–143% da mediana, uma faixa plausível.

Foi encontrada uma regra física que identifica exatamente essas 21 linhas sem utilizar o preço:

```text
tipo = Apartamento
quartos > 0
area_util / quartos > 200
```

Nenhum outro apartamento do treino foi selecionado pela regra. No teste, ela identificou somente os IDs `228`, `570` e `1245`. Suas áreas foram corrigidas respectivamente de 2.735 para 273,5 m², de 569 para 56,9 m² e de 577 para 57,7 m².

Dois erros de escala no alvo também foram considerados claros com base na mediana do bairro:

- ID `2749`: preço dividido por 10;
- ID `4316`: preço dividido por 10.

O ID `6383` teve a quantidade de vagas corrigida de 30 para 3. Cinco linhas continuaram excluídas por não possuírem uma correção suficientemente confiável: `5910`, `2405`, `4568`, `6004` e `6654`.

Com isso, o experimento utilizou 4.678 dos 4.683 imóveis. Os resultados foram:

```text
KFold aleatório: 23,60%
Desvio entre folds: 0,32 ponto percentual
Holdout com os 20% menores IDs: 22,03%
```

O KFold ficou 0,21 ponto acima da base limpa, mas apresentou uma variação muito menor. No holdout que aproxima a direção do Kaggle, melhorou os 22,31% da base quase completa sem correção e os 25,09% da base limpa.

O script `src/gerar_submission_dados_corrigidos.py` gerou `submissions/submission_lightgbm_categorico_800_dados_corrigidos.csv`. O arquivo contém 2.000 IDs únicos, nenhum valor ausente e nenhum preço negativo. As correções permanecem isoladas nesse experimento até serem confirmadas pelo Kaggle; o CSV original e o pipeline principal não foram alterados.

No Kaggle, essa versão obteve 25,11%, melhor resultado do projeto até esse momento. O ganho confirma que as correções de escala eram úteis, embora o resultado público ainda tenha ficado acima dos 22,03% do holdout por IDs baixos.

## Correspondências estruturais de imóveis repetidos

Como 433 imóveis do teste possuem uma combinação completa de features também encontrada no treino, foi avaliado um pós-processamento que utiliza a mediana de preço de registros correspondentes. A correspondência é aprendida somente no treino de cada fold, evitando utilizar o preço da validação.

Três chaves foram comparadas. A melhor foi a estrutural:

```text
tipo, bairro, area_util, area_extra, quartos, suites, vagas
```

Para imóveis com correspondência, a previsão final utiliza 75% da previsão do LightGBM e 25% da mediana dos registros estruturais. O peso reduz o risco de substituir o modelo por uma mediana de anúncios apenas parcialmente equivalentes.

Os resultados de `src/analisar_correspondencias_repetidas.py` foram:

```text
KFold sem correspondência: 23,60%
KFold com correspondência: 23,35%

Holdout por ID sem correspondência: 22,03%
Holdout por ID com correspondência: 21,84%
```

Na base final, 988 dos 2.000 imóveis de teste encontraram pelo menos uma correspondência estrutural no treino. Foi gerada a candidata `submissions/submission_lightgbm_corrigido_correspondencia_25.csv`, validada sem IDs duplicados, valores ausentes ou preços negativos.

No Kaggle, essa candidata obteve 24,78%, melhorando 0,33 ponto percentual sobre os 25,11% do mesmo modelo sem correspondências. O ganho público foi coerente e até superior aos ganhos internos de 0,19–0,25 ponto, confirmando que anúncios ou registros repetidos são relevantes no conjunto de teste.

### Variação do peso das correspondências

Para verificar se os 25% usados na melhor submission pública estavam próximos do ponto ideal, foram geradas quatro variações mantendo todos os outros dados, correções, features e parâmetros iguais:

| Peso da correspondência | KFold | Holdout por ID | Arquivo |
| ---: | ---: | ---: | :--- |
| 10% | 23,46% | 21,92% | `submission_correspondencia_10.csv` |
| 20% | 23,37% | 21,86% | `submission_correspondencia_20.csv` |
| 25% | 23,35% | 21,84% | Referência pública de 24,78% |
| 30% | **23,34%** | **21,84%** | `submission_correspondencia_30.csv` |
| 40% | 23,35% | 21,88% | `submission_correspondencia_40.csv` |

O peso de 30% foi o melhor internamente, mas sua vantagem sobre 25% é mínima. Os arquivos de 20% e 30% diferem da referência em média cerca de R$ 2,1 mil por imóvel; os arquivos de 10% e 40% diferem cerca de R$ 6,2 mil. Todos possuem 2.000 IDs únicos, nenhum valor ausente e nenhum preço negativo.

No Kaggle, o peso de 30% obteve 24,76% e o peso de 40% obteve 24,77%, contra 24,78% do peso de 25%. Os três resultados formam um platô entre 25% e 40%; 30% passa a ser a referência, mas a diferença de apenas 0,01–0,02 ponto indica que refinar o peso provavelmente traria mais ajuste ao leaderboard do que ganho generalizável.

### Árvores e taxa de aprendizado com correspondência de 30%

Com os dados e o peso de correspondência congelados, foram criadas quatro configurações ao redor das 800 árvores e `learning_rate=0,03` da referência. O produto aproximado entre árvores e taxa foi mantido constante para testar principalmente a granularidade do boosting.

| Árvores | `learning_rate` | KFold | Holdout por ID | Arquivo |
| ---: | ---: | ---: | ---: | :--- |
| 400 | 0,06 | 23,30% | 21,93% | `submission_arvores_400_lr006_correspondencia_30.csv` |
| 600 | 0,04 | **23,24%** | **21,85%** | `submission_arvores_600_lr004_correspondencia_30.csv` |
| 800 | 0,03 | 23,34% | 21,84% | Referência pública de 24,76% |
| 1.200 | 0,02 | 23,32% | 22,02% | `submission_arvores_1200_lr002_correspondencia_30.csv` |
| 1.600 | 0,015 | 23,29% | 21,91% | `submission_arvores_1600_lr0015_correspondencia_30.csv` |

A configuração de 600 árvores foi a única que trouxe uma melhora clara no KFold e praticamente empatou no holdout por ID. As quatro submissions possuem 2.000 IDs únicos, nenhum valor ausente e nenhum preço negativo. Os resultados completos estão em `resultados/comparacao_arvores_correspondencia_30.csv`.

Outra anomalia ainda não corrigida foi identificada em `area_extra`: apartamentos apresentam valores como 4.173 m² no treino e 6.022, 2.810 e 2.461 m² no teste. Esses valores provavelmente representam 41,73, 60,22, 28,10 e 24,61 m², mas essa hipótese deve ser testada separadamente para não misturar mudanças.

## Correção de `area_extra` em apartamentos

A hipótese foi testada isoladamente com a seguinte regra:

```text
tipo = Apartamento
area_extra > 1000
area_extra corrigida = area_extra / 100
```

A regra corrigiu os IDs `3656` e `6654` no treino e os IDs `1012`, `1969` e `1998` no teste. Casas com terrenos grandes não foram alteradas. Como a correção tornou plausíveis os 174,5 m² de área extra do ID `6654`, essa linha voltou ao treinamento; permaneceram excluídos apenas `5910`, `2405`, `4568` e `6004`.

Mantendo o LightGBM de 800 árvores e a correspondência estrutural com peso de 25%, os resultados foram:

```text
KFold antes da correção: 23,35%
KFold depois da correção: 23,11%

Holdout por ID antes: 21,84%
Holdout por ID depois: 21,84%
```

A correção melhorou a validação aleatória sem prejudicar o holdout por ID. O script `src/gerar_submission_area_extra_corrigida.py` gerou `submissions/submission_lightgbm_area_extra_corrigida_correspondencia_25.csv`, com 2.000 IDs únicos, nenhum valor ausente e nenhum preço negativo.

No Kaggle, essa versão obteve 25,19%, piorando 0,41 ponto percentual em relação aos 24,78% da versão sem a correção de `area_extra`. A hipótese foi rejeitada e a submission de 24,78% permanece como referência. Isso mostra que plausibilidade física e melhora interna não são suficientes para justificar uma alteração quando o padrão de armazenamento do conjunto oculto pode ser diferente.

Uma auditoria posterior confirmou que não existem valores ausentes, IDs duplicados, preços não positivos ou imóveis com mais suítes que quartos. Foram encontrados 430 grupos de features completamente repetidas no treino, envolvendo 1.061 linhas. Entretanto, 226 grupos apresentam amplitude de preço superior a 10% da mediana, 63 superam 30% e dois superam 100%. Por isso, a mediana de correspondências deve continuar suavizada e não deve substituir integralmente o modelo.

## Limitações atuais

- A busca cobriu somente 40 combinações e não esgota o espaço possível de hiperparâmetros.
- O XGBoost ainda utiliza apenas sua configuração inicial.
- A melhor configuração foi escolhida nos mesmos folds usados para comparar as combinações; uma validação repetida ou aninhada daria uma estimativa ainda mais conservadora.
- O resultado público do Kaggle também pode variar conforme a composição do conjunto oculto.
- O limite de 10 imóveis para bairros raros ainda precisa ser confirmado no conjunto oculto.

## Geração da primeira submission

Depois da comparação interna, o LightGBM com log do preço foi escolhido para gerar a primeira submission por ter apresentado o menor RMSPE da validação atual. Foi criado o script `src/gerar_submission.py`, separado do arquivo de experimentos, para deixar clara a diferença entre avaliar um modelo e produzir previsões finais.

Na modelagem de validação, o modelo foi treinado com 80% dos dados e avaliado nos 20% restantes. Na geração da submission, um novo `LGBMRegressor` foi criado com os mesmos hiperparâmetros e treinado com todos os 4.654 imóveis disponíveis. Não existe conjunto de validação nessa etapa, pois o objetivo é aproveitar todos os dados rotulados depois que a configuração já foi avaliada.

O conjunto de teste contém 2.000 imóveis e não possui a coluna `preco`. Ele passa pela mesma função `criar_features_modelo` utilizada no treino, garantindo que as features numéricas, o tratamento das categorias e o one-hot encoding sejam aplicados de forma consistente.

Como treino e teste podem possuir bairros ou tipos diferentes, o one-hot encoding feito separadamente pode produzir conjuntos de colunas distintos. Para corrigir isso, `x_teste` é reorganizado com `reindex(columns=x.columns, fill_value=0)`. Assim:

- as colunas ficam na mesma ordem usada durante o treinamento;
- colunas existentes no treino e ausentes no teste são criadas com valor zero;
- categorias exclusivas do teste são descartadas, pois o modelo não aprendeu coeficientes ou divisões para elas.

O modelo final aprende `log1p(preco)`. Suas previsões são produzidas nessa escala e convertidas novamente para reais com `expm1`. Os valores são arredondados para duas casas decimais antes de serem colocados na coluna `preco`.

O arquivo `data/exemplo_arquivo_respostas.csv` é utilizado como estrutura da resposta. Antes da substituição dos preços, um `assert` confirma que os IDs do exemplo são exatamente iguais e estão na mesma ordem dos IDs do conjunto de teste. O resultado é salvo em `submissions/submission.csv` com `index=False`, evitando a criação de uma coluna extra com o índice do pandas.

O fluxo final ficou:

```text
treino limpo -> criação de features -> treino final com todos os dados
teste -> mesmas features -> alinhamento das colunas -> previsão em log
previsão em log -> conversão para reais -> preenchimento do arquivo de exemplo
```

Essa primeira submission ainda utiliza os hiperparâmetros iniciais do LightGBM. Ela funciona como referência externa antes da implementação de validação cruzada e da otimização dos hiperparâmetros.

No Kaggle, essa configuração inicial obteve erro de 27,85%. O valor foi superior ao erro interno de 25,05%, confirmando que a divisão única estava otimista em relação ao conjunto oculto.

Depois da busca, o gerador foi atualizado com os melhores hiperparâmetros. A nova versão foi salva em `submissions/submission_lightgbm_ajustado.csv`, mantendo `submissions/submission.csv` como referência da configuração anterior. O novo arquivo contém 2.000 IDs únicos, nenhuma ausência e nenhum preço negativo.

No Kaggle, o LightGBM ajustado obteve aproximadamente 27,95%, uma piora de 0,10 ponto percentual em relação aos 27,85% da configuração inicial. A diferença é pequena e os modelos podem ser considerados praticamente empatados no conjunto público, mas o ganho observado na validação cruzada não se confirmou externamente. Por isso, a configuração inicial permanece como a referência atual e uma nova ampliação da busca de hiperparâmetros não será a prioridade imediata.

## Próximos experimentos

1. Adotar holdouts por ID junto com a validação aleatória para os próximos experimentos.
2. Avaliar uma regra de correspondência para os 433 imóveis de teste com features repetidas no treino.
3. Analisar a importância das features do LightGBM categórico.
4. Reavaliar individualmente apenas os seis erros extremos ainda removidos.
5. Ajustar o XGBoost usando os mesmos conjuntos de validação corrigidos.

## XGBoost inicial e blend com LightGBM

Foi criado `src/gerar_submission_xgboost_inicial.py` para testar o XGBoost sem alterar o pré-processamento da melhor submission anterior. O experimento mantém as correções já aceitas, todas as features criadas, `bairro` como categoria nativa, agrupamento de bairros com menos de 10 ocorrências, alvo em `log1p` e correspondência estrutural com peso de 30%.

O XGBoost inicial utilizou:

```text
n_estimators=800
learning_rate=0,03
max_depth=4
min_child_weight=5
subsample=0,8
colsample_bytree=0,8
reg_alpha=0,05
reg_lambda=1,0
tree_method="hist"
enable_categorical=True
```

Esses valores formam uma referência conservadora: árvores relativamente rasas, amostragem de linhas e colunas para reduzir sobreajuste e regularização moderada. Eles ainda não são resultado de uma busca de hiperparâmetros.

Também foi testado um blend simples, calculando 50% da previsão bruta do XGBoost e 50% da previsão bruta do LightGBM de 400 árvores. Depois dessa média, a correspondência estrutural de 30% foi aplicada da mesma forma aos três candidatos.

| Modelo | KFold | Desvio entre folds | Holdout por ID |
| :--- | ---: | ---: | ---: |
| XGBoost | 23,22% | 0,73 p.p. | 22,01% |
| LightGBM 400 | 23,30% | 0,56 p.p. | 21,93% |
| Blend 50/50 | **23,14%** | 0,61 p.p. | **21,86%** |

O XGBoost isolado superou levemente o LightGBM no KFold, mas foi pior no holdout por ID. O blend foi o melhor nos dois critérios, o que indica que os modelos cometem parte dos erros em imóveis diferentes. O ganho ainda é pequeno e precisa ser confirmado no Kaggle antes de ajustar os hiperparâmetros.

No Kaggle, o XGBoost isolado obteve 25,85% e o blend 50/50 obteve 25,16%. Os dois resultados ficaram abaixo da referência de aproximadamente 24,76%, embora o blend tivesse sido o melhor candidato nas duas validações internas. Portanto, essa configuração inicial do XGBoost foi rejeitada e o blend não deve substituir o LightGBM atual. O resultado também reforça que diferenças internas pequenas, na ordem de décimos de ponto percentual, não estão sendo estimativas confiáveis da ordenação no conjunto oculto.

Foram gerados:

- `submissions/submission_xgboost_inicial_correspondencia_30.csv`
- `submissions/submission_blend_xgb_lgbm400_50_correspondencia_30.csv`
- `resultados/comparacao_xgboost_inicial.csv`

## Busca ampliada do XGBoost

Depois dos resultados públicos de 25,85% para o XGBoost inicial e 25,16% para o blend, o pré-processamento foi congelado e foi realizada uma busca específica para o XGBoost. O script `src/buscar_xgboost.py` avaliou 159 configurações no holdout formado pelos 20% menores IDs do treino. Esse corte recebeu maior importância porque os IDs de teste vão de 0 a 1.999, enquanto os IDs de treino vão de 2.000 a 6.682.

A busca comparou crescimento por profundidade e por número de folhas, profundidade, peso mínimo dos filhos, número de árvores, taxa de aprendizado, amostragem de linhas e colunas e regularizações L1, L2 e `gamma`. Também foram comparadas três representações de bairro:

- categoria nativa do XGBoost;
- one-hot encoding;
- média suavizada do log do preço por m² do bairro, calculada somente com a parte de treino de cada corte, mais a frequência do bairro.

A terceira representação é uma codificação supervisionada. Ela foi refeita dentro de cada fold para impedir que o preço de validação entrasse na criação da própria feature. As cinco melhores configurações de cada representação, além dos melhores resultados gerais, passaram depois pelo KFold de cinco divisões. Quinze finalistas receberam essa validação completa.

O bairro categórico nativo continuou sendo a melhor representação. One-hot e a codificação por preço por m² ficaram atrás. `max_depth=4` também permaneceu em todos os melhores candidatos. O peso de correspondência de 30% foi melhor que 0%, 15% e 45% em todos os principais finalistas.

Foram escolhidas cinco submissions:

| Ordem | Candidato | Bairro | Árvores/taxa | Alteração principal | Bloco por ID | KFold |
| ---: | :--- | :--- | :--- | :--- | ---: | ---: |
| 1 | Equilibrado | Nativo | 800 / 0,03 | `min_child_weight=12`, L1=0,2 e L2=3 | 21,77% | **23,06%** |
| 2 | Gamma | Nativo | 800 / 0,03 | `min_child_weight=1` e `gamma=0,1` | 21,76% | 23,09% |
| 3 | Mais árvores | Nativo | 1.200 / 0,02 | `min_child_weight=12` | 21,78% | 23,12% |
| 4 | Bloco por ID | Nativo | 300 / 0,08 | configuração mais forte no corte próximo do teste | **21,67%** | 23,53% |
| 5 | Bairro por preço/m² | Target encoding | 800 / 0,03 | representação numérica supervisionada do bairro | 22,06% | 23,24% |

Os arquivos são, respectivamente:

- `submission_xgb_01_equilibrado_corr30.csv`
- `submission_xgb_02_gamma_corr30.csv`
- `submission_xgb_03_1200_arvores_corr30.csv`
- `submission_xgb_04_300_arvores_bloco_id_corr30.csv`
- `submission_xgb_05_bairro_target_m2_corr30.csv`

Todos possuem 2.000 IDs únicos, nenhum valor ausente e nenhum preço não positivo. A ordem sugerida de envio é 1, 4, 2, 3 e 5: primeiro o candidato mais equilibrado, depois o melhor no corte por ID, em seguida as duas alternativas de regularização e por último a representação experimental de bairro. Os resultados completos estão em `resultados/busca_xgboost_estruturas.csv`, `resultados/busca_xgboost_holdout.csv`, `resultados/busca_xgboost_finalistas.csv` e `resultados/xgboost_submissions_escolhidas.csv`.

Os resultados no Kaggle foram:

| Candidato | Kaggle |
| :--- | ---: |
| 01 - Equilibrado | 25,21% |
| 02 - Gamma | 25,80% |
| 03 - 1.200 árvores | 25,52% |
| 04 - 300 árvores, bloco por ID | 25,66% |
| 05 - Bairro por preço/m² | **24,86%** |

O candidato 01 melhorou bastante os 25,85% do XGBoost inicial, mostrando que regularização e `min_child_weight` eram relevantes. Entretanto, o principal resultado foi a inversão entre os candidatos 01 e 05: a codificação de bairro por preço por m² era pior no KFold e no holdout por ID, mas foi 0,35 ponto percentual melhor no Kaggle. O candidato 04, escolhido por ser o melhor no bloco de IDs baixos, também não confirmou sua vantagem externamente. Portanto, nem o KFold aleatório nem um único bloco de IDs estão ordenando corretamente pequenas diferenças entre os modelos. A representação de localização parece ter mais potencial público que novos ajustes finos das árvores do XGBoost.

## Pesos por preço e aproximação do RMSPE

Como o RMSPE penaliza o erro relativo, foi testado `sample_weight = 1 / preco^alpha`, normalizado para média 1. O alvo continuou sendo `log1p(preco)` e as funções objetivo permaneceram `regression` no LightGBM e `reg:squarederror` no XGBoost. Portanto, o experimento não substitui a loss por RMSPE exato: ele aumenta progressivamente a importância dos imóveis baratos sobre a aproximação percentual já produzida pelo alvo em log.

O script `src/testar_alphas_sample_weights.py` comparou sete valores de alpha nos mesmos cinco folds, mantendo as correções de dados, categorias, hiperparâmetros e correspondência estrutural de 30%. O resultado do blend 50/50 foi:

| Alpha | RMSPE KFold | Desvio | Holdout por ID |
| ---: | ---: | ---: | ---: |
| 0,00 | 23,14% | 0,61 p.p. | 21,86% |
| 0,25 | 22,77% | 0,52 p.p. | 21,75% |
| 0,50 | 22,32% | 0,69 p.p. | 21,42% |
| 0,75 | 21,98% | 0,73 p.p. | 21,11% |
| 1,00 | 21,73% | 0,75 p.p. | 20,96% |
| 1,25 | 21,54% | 0,66 p.p. | **20,85%** |
| 1,50 | **21,54%** | 0,70 p.p. | 20,91% |

O ganho veio principalmente das três faixas inferiores de preço. No quartil de até R$ 355 mil, o RMSPE do blend caiu de 27,34% para 23,78% entre alpha 0 e 1,50. Em contrapartida, no quartil mais caro ele subiu de 22,62% para 23,77%. Isso confirma que os pesos alteram de fato a prioridade do modelo, em vez de melhorar todas as faixas simultaneamente.

Como alpha 1,50 estava no limite superior da primeira grade, alpha 1,75 também foi avaliado. Ele chegou a 21,52% no KFold original, mas piorou o holdout por ID para 21,20%. Os três finalistas passaram então por KFold repetido com sementes 7, 42 e 2026, totalizando 15 folds por configuração:

| Modelo | Alpha 1,25 | Alpha 1,50 | Alpha 1,75 |
| :--- | ---: | ---: | ---: |
| LightGBM | 21,75% | 21,68% | **21,65%** |
| XGBoost | 21,63% | **21,55%** | 21,59% |
| Blend 50/50 | 21,58% | 21,51% | **21,50%** |

A vantagem do blend com alpha 1,75 sobre 1,50 foi de apenas 0,005 ponto percentual nos 15 folds, enquanto o holdout por ID piorou 0,29 ponto. Alpha 1,50 foi escolhido como compromisso: praticamente empatado no KFold repetido, melhor para o XGBoost e mais estável no corte por ID. Alpha 1,25 e 1,75 não geraram arquivos separados porque as previsões dos alphas próximos são muito correlacionadas e não justificam consumir mais tentativas no Kaggle.

Foram geradas somente duas submissions:

- `submission_lightgbm_sample_weight_a150_corr30.csv`
- `submission_blend_50_sample_weight_a150_corr30.csv`

Ambas contêm 2.000 IDs únicos, nenhum valor ausente e nenhum preço não positivo. No Kaggle, o LightGBM com alpha 1,5 obteve 22,19% e o blend XGBoost/LightGBM com alpha 1,5 obteve 21,86%. A versão anterior do blend com alpha 1 obteve 22,63%, confirmando externamente a vantagem dos pesos mais agressivos. Os detalhes estão em `resultados/comparacao_alphas_sample_weights.csv`, `resultados/rmspe_alphas_por_faixa_preco.csv` e nos arquivos `resultados/alphas_finalistas_*`.

## CatBoost com categorias nativas e pesos por preço

O CatBoost 1.2.10 foi testado como modelo isolado, sem correspondências estruturais e sem blend com LightGBM ou XGBoost. O script `src/testar_catboost_alphas.py` preserva as correções de dados e as features numéricas do pipeline, mas mantém `tipo`, `bairro`, `tipo_vendedor` e `diferenciais` como categorias nativas. Dessa forma, não é necessário aplicar one-hot encoding ou agrupar previamente os bairros raros.

A configuração inicial utilizou:

```text
loss_function=RMSE
iterations=1000
learning_rate=0,03
depth=6
l2_leaf_reg=5
random_strength=1
bootstrap_type=Bayesian
bagging_temperature=1
boosting_type=Ordered
```

O alvo continuou em `log1p(preco)`. Foram comparados os pesos `1 / preco^alpha` para alpha 0, 1 e 1,5 nos mesmos cinco folds e no holdout dos 20% menores IDs:

| Alpha | RMSPE KFold | Desvio | RMSPE de treino | Holdout por ID |
| ---: | ---: | ---: | ---: | ---: |
| 0,0 | 24,32% | 0,26 p.p. | 22,02% | 23,95% |
| 1,0 | **22,68%** | 0,37 p.p. | **20,58%** | **22,51%** |
| 1,5 | 22,81% | 0,48 p.p. | 21,15% | 22,58% |

Alpha 1 foi o melhor nos dois critérios. Em relação ao CatBoost sem pesos, reduziu o RMSPE médio em 1,64 ponto percentual e o holdout por ID em 1,44 ponto. Alpha 1,5 aumentou ainda mais a prioridade dos baratos, mas passou do melhor equilíbrio: perdeu 0,13 ponto no KFold e 0,06 ponto no holdout em relação a alpha 1.

A análise por quartis confirmou o deslocamento. Entre alpha 0 e 1,5, o RMSPE do quartil de até R$ 355 mil caiu de 28,38% para 22,26%, enquanto o quartil acima de R$ 830 mil piorou de 24,72% para 28,08%. Alpha 1 interrompe parte dessa deterioração nos imóveis caros e por isso obteve o melhor resultado total.

Nenhuma submission foi gerada nesta etapa. Os resultados completos estão em `resultados/catboost_alphas_resumo.csv`, `resultados/catboost_alphas_folds.csv`, `resultados/catboost_alphas_faixas_preco.csv` e `resultados/catboost_alphas_correlacao_erros.csv`.

## Correspondências aplicadas ao CatBoost isolado

Depois da comparação dos alphas, o script `src/testar_catboost_correspondencias.py` testou os CatBoost com alpha 1 e 1,5 usando pesos de correspondência de 0%, 15% e 30%. Nenhum outro modelo participou das previsões. A mediana de correspondência foi calculada somente com o treino de cada fold e combinada posteriormente com a previsão do CatBoost, evitando vazamento de alvo.

| Alpha | Correspondência | RMSPE KFold | Desvio | Holdout por ID |
| ---: | ---: | ---: | ---: | ---: |
| 1,0 | 0% | 22,68% | 0,42 p.p. | 22,51% |
| 1,0 | 15% | 22,24% | 0,45 p.p. | 22,17% |
| 1,0 | 30% | 21,97% | 0,51 p.p. | 22,00% |
| 1,5 | 0% | 22,81% | 0,54 p.p. | 22,58% |
| 1,5 | 15% | 22,27% | 0,56 p.p. | 22,16% |
| 1,5 | 30% | **21,94%** | 0,60 p.p. | **21,92%** |

A correspondência de 30% melhorou os cinco folds para os dois alphas. Ela reduziu 0,71 ponto percentual do CatBoost com alpha 1 e 0,87 ponto do CatBoost com alpha 1,5. No holdout, os ganhos foram de 0,52 e 0,66 ponto, respectivamente. A cobertura foi de 45,96% nas previsões out-of-fold e 43,59% no holdout por ID.

Nos 2.150 imóveis OOF com correspondência, alpha 1,5 passou de 20,39% para 18,19% com peso de 30%. As 2.528 linhas sem correspondência permanecem inalteradas por definição. A melhora não foi uniforme entre preços: a correspondência ajudou as três faixas superiores, mas piorou ligeiramente o quartil de até R$ 355 mil. Com alpha 1,5, esse quartil passou de 22,26% para 22,67%, enquanto o quartil acima de R$ 830 mil melhorou de 28,08% para 26,11%.

Alpha 1,5 com 30% foi a melhor combinação global, mas sua vantagem sobre alpha 1 com 30% foi pequena: 0,04 ponto no KFold e 0,08 ponto no holdout. Os dois continuam úteis para uma futura análise por regime, pois alpha 1,5 permanece melhor nos baratos e alpha 1 é consideravelmente mais seguro nos imóveis caros. Nenhuma submission ou blend entre modelos foi gerado.

Além dos resumos, foram salvas as previsões brutas em `resultados/catboost_correspondencias_oof.csv` e `resultados/catboost_correspondencias_holdout_id.csv`. Elas permitem testar posteriormente roteamento ou blend sem repetir o treinamento do CatBoost.

## Submissions CatBoost com correspondência de 30%

Depois da validação, o script `src/gerar_submissions_catboost_correspondencias.py` treinou dois CatBoost finais nas 4.678 linhas rotuladas. Os modelos mantiveram as 1.000 iterações e diferem somente no sample weight: alpha 1 e alpha 1,5. Nos 988 imóveis de teste com correspondência estrutural, equivalentes a 49,40% do conjunto, a previsão final combina 70% do CatBoost com 30% da mediana correspondente.

Foram gerados:

- `submissions/submission_catboost_a100_corr30.csv`
- `submissions/submission_catboost_a150_corr30.csv`

Os dois arquivos contêm 2.000 IDs únicos, nenhum valor ausente e nenhum preço não positivo. As previsões finais possuem correlação de 0,9964, mas alpha 1,5 produz preços em média 2,21% menores que alpha 1, comportamento coerente com a maior ênfase nos imóveis baratos. As previsões brutas e corrigidas dos dois modelos foram preservadas em `resultados/catboost_previsoes_teste_alphas_corr30.csv` para uso posterior sem novo treinamento.

Como alpha 1,5 com 30% obteve o menor RMSPE interno, sua submission foi a primeira candidata. No Kaggle, o CatBoost com alpha 1 obteve 22,34% e o CatBoost com alpha 1,5 obteve 21,82%, confirmando que a ponderação mais agressiva também era superior externamente.

## Blend global CatBoost e árvores com alpha 1,5

O script `src/gerar_blend_global_catboost_arvores.py` combinou 60% da submission CatBoost com alpha 1,5 e 40% da submission que já mistura igualmente XGBoost e LightGBM com alpha 1,5. A composição efetiva é, portanto, 60% CatBoost, 20% XGBoost e 20% LightGBM. Todos os componentes já utilizavam correspondência estrutural de 30%.

O resultado foi salvo em `submissions/submission_blend_global_cat60_xgb20_lgb20_a150_corr30.csv`. O arquivo contém 2.000 IDs únicos, nenhum valor ausente e nenhum preço não positivo. No Kaggle, obteve 21,45%, melhorando 0,37 ponto sobre o CatBoost alpha 1,5 e 0,41 ponto sobre o blend XGBoost/LightGBM alpha 1,5. Esse resultado confirmou que os modelos cometem erros complementares e passou a ser a referência generalista.

## Busca do CatBoost especialista em imóveis baratos

O blend global de 60% CatBoost e 40% XGBoost/LightGBM obteve 21,45% no Kaggle. Depois desse resultado, foi realizada uma busca específica por um CatBoost especialista nos imóveis baratos. O melhor candidato utiliza alvo em log, `sample_weight=1/preco^2`, profundidade 7, 600 árvores, `random_strength=0`, bootstrap Bayesiano e nenhuma correspondência. Uma calibração multiplicativa cross-fit reduziu seu RMSPE OOF nos 1.178 imóveis de até R$ 355 mil para 18,90%, contra 19,29% do CatBoost barato original calibrado.

O relatório completo, incluindo hipóteses rejeitadas, curvas por preço, estabilidade pareada e limitações, está em `docs/relatorio_catboost_especialista_baratos.md`. A configuração consolidada está em `resultados/catboost_barato_melhor_config.json`.

## Juiz e blend roteado

O generalista foi reconstruído com RMSPE OOF de 21,4066%, muito próximo dos 21,45% públicos. Quatro juízes supervisionados e um corte suave pelo preço previsto foram comparados de forma cross-fit. O classificador de utilidade foi o vencedor, com 21,3394%, e também passou na auditoria por bloco de IDs: a intensidade integral reduziu o RMSPE de 21,1745% para 21,0035%.

O especialista e o juiz foram então treinados em toda a base. Foram geradas três intensidades convexas do roteamento: 0,40, 0,70 e 1,00. Os detalhes da arquitetura, validação, arquivos e interpretação dos futuros scores do Kaggle estão em `docs/relatorio_juiz_especialista.md`.

No Kaggle, essas intensidades obtiveram respectivamente 21,38%, 21,34% e 21,33%, contra 21,45% do generalista. A curva pública indicou que novos ajustes de intensidade não trariam melhora visível. Um juiz contínuo foi então treinado para estimar diretamente o peso ótimo do especialista, usando o impacto `(especialista-generalista)^2/preco^2` de cada linha no RMSPE. A variante com alvo projetado em `[0,1]` chegou a 21,2873% cross-fit e 20,9536% no holdout por IDs, superando o juiz binário em aproximadamente 0,05 ponto nos dois testes. No Kaggle, obteve 21,30%.

Um segundo juiz contínuo passou então a dividir dinamicamente a parcela generalista entre CatBoost e árvores. A versão escolhida ancora a fração média de árvores no ótimo público de 48,7%, mas permite variação por imóvel. Ela chegou a 21,2058% cross-fit, 20,7844% no holdout e melhorou os cinco folds. Foi gerada `submissions/submission_juiz_componentes_ancorado.csv`; os detalhes estão em `docs/relatorio_juiz_especialista.md`.

O segundo juiz obteve 21,19% no Kaggle. Uma última calibração, usando intensidade 1,75 em torno da referência de 40% de árvores, chegou a 21,1972% OOF e 20,7128% no holdout. Ela foi salva como `submissions/submission_juiz_componentes_intensidade175.csv`. Essa é a última candidata de ajuste de pesos antes da busca de um especialista para a faixa cara.

## Especialista para imóveis caros

Foi congelado um novo especialista composto por 72,5% XGBoost e 27,5%
LightGBM, treinado somente na região cara e com a função quadrática ponderada
para equivaler ao RMSPE. A configuração obteve 19,81% em média nas três
sementes para imóveis a partir de R$ 950 mil e 22,65% a partir de R$ 1,3 milhão.
No holdout por ID, superou com margem tanto as árvores atuais quanto o pipeline
público de 21,19% nessas faixas.

A análise por intervalos mostrou que o ganho consistente começa por volta de
R$ 1 milhão; abaixo disso o modelo não deve receber peso global. Nenhum juiz ou
submission foi criado nesta etapa. O protocolo, todas as hipóteses e os
artefatos estão em `docs/relatorio_especialista_caros.md`.

## Juiz do especialista caro

O especialista caro foi incorporado por um novo juiz LightGBM validado de forma
aninhada. Ele estima tanto a utilidade do especialista quanto a probabilidade
de o imóvel estar acima de R$ 1 milhão. A probabilidade de faixa elevada à
sexta potência protege as regiões em que o especialista extrapola mal.

O RMSPE aninhado médio em três sementes caiu de 21,2058% para 21,0307%, com
ganho nas três sementes e em 14 dos 15 folds. No holdout por ID, caiu de
20,7844% para 20,5392%. Foi gerada apenas
`submissions/submission_juiz_especialista_caros.csv`. O protocolo e a análise
por faixa estão em `docs/relatorio_juiz_caros.md`.

No Kaggle, porém, o juiz caro piorou para 22,32%. Ele foi descartado e a etapa
seguinte passou a separar erro do juiz de erro do especialista. Duas ablações
alteram somente os 100 imóveis com maior preço previsto pelo pipeline de 21,19%,
usando 50% e 100% do especialista caro. Os detalhes estão em
`docs/relatorio_gate_conservador_caros.md`.
