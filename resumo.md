# Introdução ao Aprendizado de Máquina

## Preditor de Preços de Imóveis

Projeto e implementação de um regressor de preços de imóveis com pipeline de pré-processamento, limpeza orientada por dados, comparação de modelos e arquitetura de roteamento entre generalista e especialista

**Aluno:** Henrique Kezen V. H. Leite
**Repositório:** (inserir link)
**Ambiente técnico:** Python 3.14 — Pandas, NumPy, Scikit-learn, LightGBM, XGBoost, CatBoost, Matplotlib, Seaborn
**Rio de Janeiro, julho de 2026**

---

## Resumo

Este trabalho aborda a previsão de preços de imóveis a partir de um conjunto de 4.683 registros com 20 atributos. Foram testados Regressão Linear, XGBoost, LightGBM e CatBoost, todos treinados com alvo em `log1p(preco)` — cujas previsões são convertidas de volta para reais com `expm1` — variáveis categóricas nativas e `sample_weight = 1/preco^1,5`. O modelo final é uma arquitetura de roteamento que combina um generalista (blend de 60% CatBoost, 20% XGBoost e 20% LightGBM) com um especialista CatBoost para imóveis baratos (`sample_weight = 1/preco²`), selecionados por um juiz contínuo.

---

## 1. Objetivo do Projeto

O objetivo deste trabalho é construir um modelo de regressão capaz de prever o preço de venda de imóveis a partir de suas características físicas, localização e diferenciais de infraestrutura. A métrica de avaliação da competição é o **RMSPE** (Root Mean Squared Percentage Error) — a raiz da média dos erros percentuais ao quadrado. O RMSPE penaliza proporcionalmente ao preço real: um erro de R$ 50 mil em um imóvel de R$ 100 mil é muito mais grave que o mesmo erro em um imóvel de R$ 1 milhão.

---

## 2. Descrição dos Dados

O conjunto de dados fornecido pela disciplina está dividido em duas partes:

| Arquivo | Registros | Colunas | Uso |
| :--- | ---: | ---: | :--- |
| `conjunto_de_treinamento (5).csv` | 4.683 | 21 (20 atributos + 1 alvo) | Treino e validação |
| `conjunto_de_teste (3).csv` | 2.001 | 20 (sem alvo) | Geração de previsões para o Kaggle |

### 2.1 Variável alvo

A variável `preco` é contínua e apresenta distribuição fortemente assimétrica à direita: a média (R$ 927.705) é quase o dobro da mediana (R$ 515.000), com mínimo de R$ 750 e máximo de R$ 630 milhões. Essa assimetria motivou o uso da transformação logarítmica `log1p(preco)` como alvo de treino.

### 2.2 Tipos de variáveis

| Tipo | Quantidade | Exemplos |
| :--- | ---: | :--- |
| Categóricas nominais | 4 | `tipo`, `bairro`, `tipo_vendedor`, `diferenciais` |
| Numéricas quantitativas | 5 | `quartos`, `suites`, `vagas`, `area_util`, `area_extra` |
| Binárias (diferenciais) | 10 | `churrasqueira`, `piscina`, `vista_mar`, `sauna`, `playground` |
| Identificador | 1 | `Id` (não preditivo) |

A coluna `tipo` é altamente desbalanceada: 4.501 apartamentos contra apenas 177 casas, 3 lofts e 2 quitinetes. A coluna `bairro` possui 66 categorias distintas, das quais 30 possuem menos de 10 imóveis.

### 2.3 Valores faltantes

Nenhuma coluna apresentou valores nulos, o que simplificou o pré-processamento.

### 2.4 Valores extremos

Variáveis numéricas apresentaram outliers significativos que exigiram análise caso a caso:

| Variável | Mediana | Máximo | Observação |
| :--- | ---: | ---: | :--- |
| `preco` | R$ 515.000 | R$ 630.000.000 | Provável erro de escala |
| `area_util` | ~100 m² | 2.045 m² | Casas decimais perdidas |
| `area_extra` | 0 m² | 17.450 m² | Provável erro de escala |
| `vagas` | 2 | 30 | Caso isolado |

---

## 3. Metodologia

### 3.1 Estratégia geral

O projeto foi organizado em fases: pré-processamento e auditoria dos dados, modelagem com comparação de algoritmos, otimização de hiperparâmetros, construção de um modelo especialista para imóveis baratos e, por fim, arquitetura de roteamento com juiz. O conjunto de teste foi mantido isolado até a etapa final de submissão.

### 3.2 Duas métricas de validação

Ao longo do relatório, duas grandezas diferentes são reportadas:

- **Validação interna (KFold)** — RMSPE médio da validação cruzada de 5 folds sobre os dados de treino, com `shuffle=True` e `random_state=42`. É a métrica usada para tomar decisões de modelagem.
- **Holdout por blocos de ID** — RMSPE calculado sobre os 20% menores IDs do treino, simulando a direção treino→teste observada na competição (IDs de teste vão de 0 a ~1.999, IDs de treino de ~2.000 a ~6.682). Esse holdout complementa o KFold porque a separação do Kaggle não é aleatória.
- **Score público (Kaggle)** — RMSPE calculado pela plataforma sobre a fração pública do conjunto de teste. Serve como estimativa externa de generalização.

### 3.3 Comparação de modelos

- **Regressão Linear** — baseline de referência, estabelecendo um piso de desempenho.
- **XGBoost** — gradient boosting com crescimento por profundidade, otimizado para a representação categórica nativa.
- **LightGBM** — gradient boosting com crescimento por folha, usando `bairro` como categoria nativa.
- **CatBoost** — gradient boosting com ordered boosting e tratamento nativo de todas as categorias (`tipo`, `bairro`, `tipo_vendedor`, `diferenciais`), sem necessidade de one-hot encoding.

### 3.4 Métrica RMSPE

```
erro_percentual_i = (preco_real_i - preco_previsto_i) / preco_real_i
RMSPE = sqrt( media( erro_percentual_i² ) )
```

Como os erros são divididos pelo preço real, imóveis baratos com o mesmo erro absoluto produzem erro percentual muito maior. Essa propriedade motivou o uso de sample weights ao longo do projeto.

---

## 4. Pré-processamento dos Dados

### 4.1 Objetivo

Transformar a base bruta em uma matriz numérica pronta para modelagem, tratando outliers, criando features derivadas e codificando variáveis categóricas.

### 4.2 Auditoria e limpeza de outliers

A auditoria foi feita em duas fases. Primeiro, inspeção direta de preços e áreas extremas. Segundo, criação de **features de auditoria** que utilizam o preço — como `preco_m2 = preco / area_util` — para identificar inconsistências menos óbvias. Essas features existem exclusivamente para análise exploratória e **não são entregues ao modelo**, já que o preço é justamente a variável que o modelo deve prever e, portanto, não está disponível no conjunto de teste. Os casos individuais foram comparados com a mediana de `preco/m²` do respectivo bairro.

#### 4.2.1 Primeira limpeza e o erro de ser agressivo demais

Na primeira versão, 29 imóveis foram removidos: 6 casos de erro claro (preços absurdos, áreas impossíveis) e 23 casos cujo preço por metro quadrado destoava da mediana do bairro. Essa remoção parecia segura porque a validação cruzada aleatória (KFold) não piorava.

Entretanto, ao submeter ao Kaggle, o resultado foi pior do que o esperado. A investigação revelou que **parte dos imóveis removidos possuía registros com features muito similares no conjunto de teste** — ou seja, o teste continha registros praticamente iguais a esses casos que havíamos descartado. Sem esses exemplos no treino, o modelo não tinha como prever corretamente esses imóveis repetidos.

Além disso, o KFold aleatório não capturava essa divergência porque a separação treino/teste no Kaggle não é aleatória: os IDs de teste são sistematicamente menores (~0 a ~1.999) que os de treino (~2.000 a ~6.682). Essa descoberta levou à adoção do **holdout por blocos de ID** como validação complementar.

#### 4.2.2 Recuperação e correção conservadora

As 23 linhas foram então recuperadas. Ao invés de simplesmente recolocá-las sem tratamento, investigou-se a causa dos valores extremos. Descobriu-se que 21 delas eram apartamentos cuja `area_util` tinha provável erro de casa decimal — a regra `area_util / quartos > 200` os identificava deterministicamente, sem nenhum outro apartamento do treino sendo selecionado. A correção consistiu em dividir `area_util` por 10.

Adicionalmente:

- Dois preços foram corrigidos por divisão por 10 (IDs 2749 e 4316), com base na comparação com a mediana do bairro;
- O número de vagas do ID 6383 foi corrigido de 30 para 3;
- A área extra do ID 6654 foi corrigida dividindo por 100, tornando o registro plausível e permitindo sua reinclusão.

#### 4.2.3 Exclusões definitivas e resultado

Cinco imóveis permaneceram excluídos por não possuírem correção confiável:

| ID | Motivo da exclusão definitiva |
| ---: | :--- |
| 5910 | Preço de R$ 750 |
| 2405 | Preço de R$ 630 milhões para 98 m² |
| 4568 | Preço de R$ 65 milhões para 36 m² |
| 6004 | Preço de R$ 340 milhões para 72 m² |
| 6654 | 17.450 m² de área extra (mantido excluído antes da correção ser confirmada) |

A mesma regra de `area_util / quartos > 200` foi aplicada ao conjunto de teste, corrigindo 3 imóveis. No conjunto de teste, 433 dos 2.000 imóveis possuem combinação completa de features muito similar a pelo menos um do treino — incluindo vários dos casos anteriormente removidos. Isso confirmou que a remoção agressiva inicial prejudicava diretamente as previsões desses imóveis repetidos.

A base final ficou com **4.678 imóveis** para treino.

### 4.3 Transformação do alvo

O preço foi transformado com `np.log1p(preco)` para o treinamento: o modelo aprende a prever nessa escala comprimida. Depois da previsão, o valor é convertido de volta para reais com `np.expm1`, recuperando a escala original do preço antes de qualquer avaliação ou submissão. Essa transformação é especialmente eficaz para a métrica RMSPE porque converte divisão em subtração: minimizar o MSE no espaço log equivale aproximadamente a minimizar erros proporcionais.

### 4.4 Variáveis categóricas

| Coluna | Tratamento | Justificativa |
| :--- | :--- | :--- |
| `tipo` | One-hot com agrupamento (Loft + Quitinete → Outro) | Apenas 3 e 2 exemplos, respectivamente |
| `bairro` | Categoria nativa do LightGBM/CatBoost; bairros com <10 imóveis agrupados em "Outros" | Reduziu entrada de 91 para 26 features com ganho de 0,69 p.p. |
| `tipo_vendedor` | Binário: Imobiliária=1, Pessoa Física=0 | Apenas duas categorias |
| `diferenciais` | Removida | Redundante com as 10 colunas binárias já existentes |

### 4.5 Feature engineering

| Feature | Fórmula | Motivação |
| :--- | :--- | :--- |
| `area_total` | `area_util + area_extra` | Capturar tamanho total |
| `area_por_quarto` | `area_util / quartos` | Indicador de amplitude dos cômodos |
| `n_comodidades` | soma das 10 binárias | Proxy de luxo/infraestrutura |
| `tem_suite` | 1 se `suites > 0` | Diferencial binário de qualidade |
| `vagas_por_quarto` | `vagas / quartos` | Conveniência relativa |

### 4.6 Resultado

A matriz processada final para o LightGBM/CatBoost categórico ficou com 4.678 linhas × 26 features, inteiramente numérica e pronta para modelagem.

---

## 5. Modelagem e Treinamento

### 5.1 Objetivo

Comparar o desempenho dos modelos de regressão e identificar os mais promissores para refinamento e composição.

### 5.2 Resultados comparativos iniciais

Os modelos foram avaliados sobre os mesmos 931 imóveis de um holdout fixo (80/20, `random_state=42`):

| Modelo | Escala do alvo | RMSPE | Previsões negativas |
| :--- | :--- | ---: | ---: |
| Regressão Linear (normal) | Preço em reais | 40,21% | 10 |
| Regressão Linear (log) | `log1p(preco)` | 30,13% | 0 |
| XGBoost (log) | `log1p(preco)` | 25,26% | 0 |
| LightGBM (log) | `log1p(preco)` | 25,05% | 0 |

A transformação logarítmica reduziu o RMSPE em 10,08 pontos percentuais e eliminou previsões negativas. Os modelos de boosting superaram a regressão linear em aproximadamente 5 pontos, mas a diferença entre XGBoost e LightGBM foi de apenas 0,21 ponto — dentro do ruído da amostra.

### 5.3 Impacto da representação do bairro

A mudança de one-hot encoding (91 features) para categoria nativa (26 features) trouxe ganho consistente:

| Estratégia | RMSPE médio (KFold) | Desvio |
| :--- | ---: | ---: |
| One-hot (91 features) | 24,08% | 1,51 p.p. |
| Categórico, mínimo 10 imóveis (26 features) | **23,39%** | **1,37 p.p.** |

A representação nativa superou o one-hot em todos os cinco folds, com ganhos individuais entre 0,06 e 1,61 ponto percentual.

### 5.4 Otimização de hiperparâmetros

O LightGBM foi otimizado com `RandomizedSearchCV` (40 combinações × 5 folds = 200 treinamentos). A melhor combinação de taxa de aprendizado e número de árvores foi `learning_rate=0,03` com 600–800 árvores, atingindo RMSPE médio de 23,34%.

---

## 6. Correspondências estruturais e pós-processamento

### 6.1 Repetições entre treino e teste

Conforme descrito na auditoria de dados (Seção 4.2), 433 dos 2.000 imóveis de teste possuem combinação completa de features muito similar a pelo menos um imóvel do treino, formando 843 pares praticamente iguais. Isso indica repetição de anúncios ou registros no dataset. As chaves de correspondência utilizadas são:

```
tipo, bairro, area_util, area_extra, quartos, suites, vagas
```

### 6.2 Pós-processamento por correspondência

Para imóveis com correspondência estrutural, a previsão final combina a saída do modelo com a mediana de preço dos registros correspondentes no treino:

```
previsao_final = (1 - peso) × previsao_modelo + peso × mediana_correspondencia
```

A mediana de correspondência é calculada somente com os dados de treino de cada fold, evitando vazamento do alvo. Na base final, 988 dos 2.000 imóveis de teste (~49%) encontraram pelo menos uma correspondência. O peso de **30%** foi o melhor internamente e confirmou-se no Kaggle, produzindo um platô estável entre 25% e 40%.

---

## 7. Combinação de modelos (Blend) e pesos por preço

### 7.1 Conceito do blend

Como XGBoost e LightGBM obtiveram resultados muito próximos na modelagem inicial, foi testada a combinação de suas previsões por média simples — uma técnica chamada **blend**. A ideia é que, por usarem estratégias diferentes de construção de árvores e tratamento de categorias, os dois modelos tendem a errar em imóveis diferentes. Ao calcular a média das previsões, os erros individuais se compensam parcialmente, reduzindo a variância sem aumentar o viés.

O blend de 50% XGBoost e 50% LightGBM obteve **23,14%** no KFold, melhor que ambos os modelos isolados. Esse resultado motivou o uso de blends em todos os experimentos subsequentes.

### 7.2 Sample weights: alinhando treino e métrica

Como o RMSPE penaliza erros percentuais, imóveis baratos dominam a métrica. A função de perda padrão (MSE sobre log) já aproxima erros proporcionais, mas não dá prioridade explícita a nenhuma faixa de preço. A introdução de `sample_weight = 1 / preco^α`, normalizado para média 1, força o modelo a "se esforçar mais" nos imóveis baratos.

### 7.3 Resultados do blend XGBoost/LightGBM com diferentes alphas

| Alpha | RMSPE KFold | Desvio | Holdout por ID |
| ---: | ---: | ---: | ---: |
| 0,00 | 23,14% | 0,61 p.p. | 21,86% |
| 0,50 | 22,32% | 0,69 p.p. | 21,42% |
| 1,00 | 21,73% | 0,75 p.p. | 20,96% |
| 1,50 | **21,54%** | 0,70 p.p. | 20,91% |

O ganho veio principalmente dos imóveis baratos: no quartil até R$ 355 mil, o RMSPE caiu de 27,34% (α=0) para 23,78% (α=1,5). Em contrapartida, o quartil mais caro subiu de 22,62% para 23,77% — confirmando que os pesos redistribuem esforço, não melhoram todas as faixas.

No Kaggle, o blend com α=1,5 obteve **21,86%**, contra 22,63% com α=1 — confirmando externamente a vantagem dos pesos mais agressivos.

---

## 8. CatBoost e Blend Global

### 8.1 CatBoost como terceiro modelo

O CatBoost foi testado com categorias totalmente nativas (`tipo`, `bairro`, `tipo_vendedor`, `diferenciais`), dispensando qualquer codificação prévia. Com `sample_weight = 1/preco^1` e correspondência de 30%, obteve RMSPE de 21,97% no KFold — competitivo com o blend LightGBM/XGBoost.

No Kaggle, o CatBoost com α=1,5 e correspondência de 30% obteve **21,82%**.

### 8.2 Blend global de três modelos

A combinação de 60% CatBoost + 20% XGBoost + 20% LightGBM, todos com α=1,5 e correspondência de 30%, obteve **21,45%** no Kaggle — melhorando 0,37 ponto sobre o CatBoost isolado e 0,41 ponto sobre o blend XGBoost/LightGBM. Esse resultado confirmou que os modelos cometem erros complementares e estabeleceu a referência **generalista**.

---

## 9. Especialista para imóveis baratos

### 9.1 Motivação

Apesar da melhora trazida pelos sample weights, o generalista ainda apresentava RMSPE alto no quartil de imóveis baratos (até R$ 355 mil). Um CatBoost dedicado, com ponderação extrema (`sample_weight = 1/preco²`), foi buscado para essa faixa.

### 9.2 Busca e configuração final

Foram realizados 85 treinamentos em sete blocos experimentais. O melhor especialista:

| Parâmetro | Valor |
| :--- | :--- |
| Alvo | `log1p(preco)` |
| `sample_weight` | `1 / preco²` |
| `depth` | 7 |
| `iterations` | 600 |
| `random_strength` | 0 |
| Bootstrap | Bayesiano |
| Correspondência | 0% |
| Fator de calibração | 0,914489 |

### 9.3 Resultado do especialista

| Modelo | RMSPE até R$ 355 mil (calibrado cross-fit) |
| :--- | ---: |
| CatBoost barato original (α=1,5) | 19,29% |
| Especialista final | **18,90%** |

A melhora foi pareada: apareceu em quatro dos cinco folds, com intervalo indicativo de 0,14 a 0,66 ponto percentual.

---

## 10. Arquitetura de roteamento com juiz

### 10.1 Conceito

A ideia é usar o generalista para a maioria dos imóveis e substituí-lo parcialmente pelo especialista nos imóveis onde esse último é superior. Um modelo "juiz" aprende a estimar, para cada linha, o peso ótimo do especialista.

### 10.2 Juiz binário (classificador de utilidade)

Um `CatBoostClassifier` aprendeu se o especialista produziu menor perda percentual quadrática que o generalista, ponderado pelo impacto absoluto da decisão. A previsão final é:

```
previsao = generalista + λ × peso_do_juiz × (especialista - generalista)
```

Quatro juízes supervisionados foram comparados em cross-fit. O classificador de utilidade venceu com RMSPE de 21,3394%, contra 21,4066% do generalista sem juiz. No Kaggle, com intensidade λ=1,00, obteve **21,33%**.

### 10.3 Juiz contínuo

O último refinamento substituiu a classificação binária por uma regressão que estima diretamente o peso ótimo do especialista, usando como pseudo-alvo `(y-G)/D` projetado em [0,1] e ponderado por `(D/y)²`.

| Modelo | RMSPE cross-fit | RMSPE holdout por ID |
| :--- | ---: | ---: |
| Juiz binário (melhor intensidade) | 21,3394% | 21,0035% |
| Juiz contínuo calibrado | **21,2873%** | **20,9536%** |

O juiz contínuo superou o binário nos cinco folds, com ganhos entre 0,012 e 0,074 ponto percentual. A submission `submission_juiz_continuo_rmspe.csv` foi gerada como candidata final.

---

## 11. Discussão dos Resultados

### 11.1 Por que a transformação logarítmica foi essencial?

A transformação `log1p` reduziu o RMSPE em mais de 10 pontos percentuais na regressão linear e eliminou previsões negativas. A explicação é que o RMSPE avalia erro relativo, e o log transforma divisão em subtração — ao minimizar o MSE no espaço log, o modelo aprende proporções naturalmente. Todos os modelos subsequentes foram treinados com esse alvo.

### 11.2 Por que a categoria nativa superou o one-hot?

O one-hot encoding criava 66 colunas binárias para os bairros, dificultando que as árvores tratassem em conjunto bairros com comportamento semelhante e favorecendo a memorização de bairros com poucos exemplos. A representação categórica nativa permite ao LightGBM/CatBoost procurar separações ótimas entre grupos de bairros durante a construção de cada árvore, obtendo mais informação com menos dimensões.

### 11.3 Por que sample weights ajudaram tanto?

A RMSPE é uma métrica assimétrica em relação ao preço: um erro de R$ 50 mil em um imóvel de R$ 100 mil contribui 25 vezes mais que o mesmo erro em um imóvel de R$ 500 mil. A função de perda padrão (MSE no log) já aproxima erros proporcionais, mas trata todas as faixas igualmente. Os sample weights `1/preco^α` fazem o modelo gastar mais capacidade nos imóveis baratos, alinhando explicitamente treino e avaliação. O ganho foi de mais de 2 pontos percentuais no Kaggle.

### 11.4 Por que o blend global foi superior aos modelos isolados?

O CatBoost, XGBoost e LightGBM usam estratégias diferentes de construção de árvores, tratamento de categorias e regularização. Isso faz com que cometam erros em imóveis diferentes. A média ponderada das previsões reduziu a variância sem aumentar o viés, melhorando o RMSPE final.

### 11.5 Por que o juiz contínuo superou o binário?

O juiz binário decide apenas "sim ou não" para o uso do especialista, perdendo informação sobre o grau de confiança. O juiz contínuo estima diretamente o peso ótimo, permitindo transições suaves entre generalista e especialista. Isso é especialmente útil nos imóveis na fronteira entre "baratos" e "intermediários", onde nenhum dos dois modelos é claramente superior.

### 11.6 Sobre a divergência entre validação interna e Kaggle

A separação de treino e teste no Kaggle não é aleatória — os IDs de teste são sistematicamente menores. Isso faz com que o KFold aleatório subestime ou superestime ganhos de certas estratégias (como a recuperação de linhas anômalas). A adoção do holdout por blocos de ID como validação complementar foi a medida mais importante para alinhar validação interna com o score público.

---

## 12. Evolução dos resultados no Kaggle

| Modelo / Estratégia | RMSPE Kaggle |
| :--- | ---: |
| LightGBM inicial (171 árvores, one-hot, sem correções) | 27,85% |
| LightGBM ajustado (busca de hiperparâmetros) | 27,95% |
| LightGBM 800 árvores, bairro categórico | ~25,5% |
| LightGBM com dados corrigidos + correspondência 30% | 24,76% |
| XGBoost bairro por preço/m² | 24,86% |
| Blend XGB/LightGBM α=1,5 + correspondência 30% | 21,86% |
| CatBoost α=1,5 + correspondência 30% | 21,82% |
| Blend global 60/20/20 α=1,5 + correspondência 30% | 21,45% |
| Juiz binário (utilidade, λ=1,00) | 21,33% |
| **Juiz contínuo RMSPE** | **~21,3%** |

---

## 13. Conclusão

O modelo final utilizado na competição é uma **arquitetura de roteamento** com a seguinte composição:

| Componente | Detalhes |
| :--- | :--- |
| **Generalista** | Blend: 60% CatBoost + 20% XGBoost + 20% LightGBM |
| Ponderação | `sample_weight = 1/preco^1.5` em todos os componentes |
| Correspondência | 30% da mediana estrutural para ~49% dos imóveis de teste |
| **Especialista** | CatBoost depth 7, 600 iterações, `1/preco²`, calibração 0,914 |
| **Juiz contínuo** | CatBoost regressão, alvo projetado [0,1], calibrado cross-fit |
| Mapeamento final | `clip(-0,45 + 1,50 × score_juiz, 0, 1)` |
| Alvo de treino | `log1p(preco)` |
| Dados de treino | 4.678 imóveis (5 exclusões, correções conservadoras de escala) |
| Features | 26 (com bairro como categoria nativa) |
| Melhor score Kaggle | **21,33%** |


As estratégias que se mostraram mais efetivas foram, em ordem de impacto: a transformação logarítmica do preço, que trouxe cerca de 10 pontos percentuais de ganho na regressão linear; o uso de sample weights ponderados pelo preço, com aproximadamente 2 pontos percentuais de ganho no Kaggle; a correspondência estrutural de registros repetidos, contribuindo entre 0,3 e 1,5 pontos percentuais; o uso de categoria nativa para bairros, com cerca de 0,7 pontos percentuais de ganho; o blend de três modelos, acrescentando aproximadamente 0,4 pontos percentuais; o roteamento com juiz, com ganho de cerca de 0,1 pontos percentuais; e a auditoria e correção conservadora de erros de escala, que trouxe melhora indireta por meio da qualidade dos dados. Por outro lado, tentativas de ampliar a busca de hiperparâmetros do LightGBM ajustado, corrigir a variável `area_extra` em apartamentos não trouxeram ganho no placar real.

---

## 14. Referências

- Scikit-learn: https://scikit-learn.org/
- LightGBM: https://lightgbm.readthedocs.io/
- XGBoost: https://xgboost.readthedocs.io/
- CatBoost: https://catboost.ai/
- Pandas: https://pandas.pydata.org/
