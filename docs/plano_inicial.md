# 🏠 Plano Inicial — Projeto 2: Previsão de Preços de Imóveis

## 1. Visão Geral do Problema

| Item | Detalhe |
| :--- | :--- |
| **Disciplina** | EEL891 — Introdução ao Aprendizado de Máquina (2025.02) |
| **Tipo de problema** | Regressão Multivariável |
| **Objetivo** | Prever o preço de imóveis a partir de suas características |
| **Métrica de avaliação** | RMSPE (Root Mean Squared Percentage Error) |
| **Plataforma** | Kaggle |

---

## 2. Análise Inicial dos Dados

### 2.1 Arquivos disponíveis

| Arquivo | Descrição | Tamanho |
| :--- | :--- | :--- |
| `conjunto_de_treinamento (5).csv` | Dados de treino com gabarito (preço) | 4.683 linhas × 21 colunas |
| `exemplo_arquivo_respostas.csv` | Formato de submissão ao Kaggle | 2.001 linhas (Id + preço) |



### 2.2 Colunas do Dataset (21 colunas)

#### Variáveis Categóricas (texto)
| Coluna | Valores Únicos | Observação |
| :--- | :--- | :--- |
| `tipo` | 4 (Apartamento, Casa, Loft, Quitinete) | **Altamente desbalanceado**: Apartamento = 4.501, Casa = 177, Loft = 3, Quitinete = 2 |
| `bairro` | 66 bairros | 30 bairros têm menos de 10 imóveis (raros). Top 1: Boa Viagem com 1.599 imóveis |
| `tipo_vendedor` | 2 (Imobiliária, Pessoa Física) | Imobiliária = 4.556, Pessoa Física = 127 |
| `diferenciais` | ~30+ combinações | Texto descritivo combinando vários diferenciais (ex: "piscina e churrasqueira") |

#### Variáveis Numéricas
| Coluna | Média | Mín | Máx | Observação |
| :--- | :--- | :--- | :--- | :--- |
| `quartos` | 3.0 | 1 | 9 | Distribuição concentrada entre 2 e 4 |
| `suites` | 1.4 | 0 | 6 | |
| `vagas` | 1.7 | 0 | 30 | ⚠️ Máx = 30 → possível outlier |
| `area_util` | 124 m² | 21 | 2.045 | ⚠️ Máx = 2.045 m² → possível outlier |
| `area_extra` | 13 m² | 0 | 17.450 | ⚠️ Máx = 17.450 m² → outlier extremo. 75% dos imóveis têm 0 |

#### Variáveis Binárias (0 ou 1) — Diferenciais
| Coluna | Descrição |
| :--- | :--- |
| `churrasqueira` | Tem churrasqueira? |
| `estacionamento` | Estacionamento para visitantes? |
| `piscina` | Tem piscina? |
| `playground` | Tem playground? |
| `quadra` | Quadra esportiva? |
| `s_festas` | Salão de festas? |
| `s_jogos` | Salão de jogos? |
| `s_ginastica` | Sala de ginástica? |
| `sauna` | Tem sauna? |
| `vista_mar` | Vista para o mar? |

#### Target (variável alvo)
| Coluna | Média | Mediana | Mín | Máx |
| :--- | :--- | :--- | :--- | :--- |
| `preco` | R$ 927.705 | R$ 515.000 | R$ 750 | R$ 630.000.000 |

> **⚠️ ALERTA: Outliers extremos no preço!** O preço mínimo é R$ 750 (provavelmente um erro) e o máximo é R$ 630 milhões. A média (R$ 927k) é quase o dobro da mediana (R$ 515k), o que indica que poucos imóveis muito caros estão "puxando" a média para cima. Isso reforça a importância de usar a **transformação logarítmica** no preço.

### 2.3 Dados Faltantes
**Nenhum valor nulo** em nenhuma coluna. Isso simplifica bastante o pré-processamento.

---

## 3. Plano de Pré-Processamento

### 3.1 Tratamento de Outliers
- [ ] Investigar o imóvel com preço = R$ 750 (possível erro de digitação)
- [ ] Investigar imóveis com `vagas` = 30 e `area_extra` = 17.450
- [ ] Decidir se removemos ou tratamos esses outliers (testar com e sem)

### 3.2 Transformação do Target (Preço)
- [ ] Aplicar `np.log1p(preco)` para transformar o preço em escala logarítmica
- [ ] Isso é essencial porque a métrica RMSPE avalia erro **percentual**, e o log transforma divisão em subtração, fazendo o modelo aprender proporções naturalmente

### 3.3 Variáveis Categóricas

#### `tipo` (4 categorias)
- [ ] Aplicar **One-Hot Encoding**
- [ ] Considerar agrupar Loft (3 imóveis) e Quitinete (2 imóveis) em uma categoria "Outro", pois têm pouquíssimos exemplos

#### `bairro` (66 bairros, 30 raros)
- [ ] Estratégia principal: **Target Encoding** — substituir cada bairro pela média do `log(preço)` daquele bairro (usando validação cruzada interna para evitar vazamento de dados)
- [ ] Estratégia alternativa: One-Hot Encoding apenas para os top 20-30 bairros + agrupar os demais em "Outros"
- [ ] Testar ambas e comparar o desempenho

#### `tipo_vendedor` (2 categorias)
- [ ] Transformar em binário: Imobiliaria = 1, Pessoa Física = 0 (ou vice-versa)

#### `diferenciais` (texto combinado)
- [ ] Essa coluna é redundante, pois os diferenciais já estão separados nas colunas binárias (churrasqueira, piscina, etc.)
- [ ] Verificar se existe alguma informação extra (como "copa", "frente para o mar", "esquina") que não está nas colunas binárias e extrair se necessário
- [ ] Após extrair, descartar a coluna `diferenciais`

### 3.4 Feature Engineering (Criar novas features)
- [ ] `area_total` = `area_util` + `area_extra`
- [ ] `area_por_quarto` = `area_util` / `quartos`
- [ ] `n_comodidades` = soma de todas as colunas binárias (churrasqueira + piscina + sauna + ...)
- [ ] `tem_suite` = 1 se `suites` > 0, 0 caso contrário
- [ ] `vagas_por_quarto` = `vagas` / `quartos`
- [ ] `preco_medio_bairro` = média do preço do bairro (Target Encoding)
- [ ] Possível: `eh_boa_viagem` = flag específica para Boa Viagem (bairro dominante com 34% dos dados)

### 3.5 Escalonamento
- [ ] Aplicar `StandardScaler` apenas para modelos lineares (Ridge, Lasso, ElasticNet)
- [ ] Não aplicar escalonamento para modelos de árvore (Random Forest, XGBoost, LightGBM)

---

## 4. Plano de Modelagem

### 4.1 Etapa 1 — Baseline (Modelo simples de referência)
- [ ] Treinar uma **Regressão Linear** simples com poucas features (area_util, quartos, vagas)
- [ ] Calcular o RMSPE via validação cruzada (5-fold)
- [ ] Este resultado serve apenas como **referência** para comparar com modelos melhores


### 4.3 Etapa 3 — Modelos de Árvore (foco principal)
- [ ] Treinar **Random Forest Regressor**
- [ ] Treinar **Gradient Boosting Regressor** (sklearn)
- [ ] Treinar **XGBoost** (xgboost)
- [ ] Treinar **LightGBM** (lightgbm)
- [ ] Comparar os resultados de todos via validação cruzada

### 4.4 Etapa 4 — Ajuste de Hiperparâmetros
- [ ] Para o(s) melhor(es) modelo(s), ajustar hiperparâmetros usando **Optuna** ou **RandomizedSearchCV**
- [ ] Hiperparâmetros a ajustar (exemplo para XGBoost/LightGBM):
  - `n_estimators`: [100, 300, 500, 1000]
  - `learning_rate`: [0.01, 0.05, 0.1]
  - `max_depth`: [3, 5, 7, 10]
  - `subsample`: [0.7, 0.8, 0.9, 1.0]
  - `colsample_bytree`: [0.7, 0.8, 0.9, 1.0]
  - `min_child_weight`: [1, 3, 5, 10]
  - `reg_alpha` (L1) e `reg_lambda` (L2): [0, 0.1, 1, 10]

### 4.5 Etapa 5 (Opcional) — Ensemble (Combinação de Modelos)
- [ ] Criar um **ensemble** combinando as previsões dos melhores modelos
- [ ] Técnica: média ponderada ou Stacking (usar previsões de vários modelos como features para um modelo final)

---

## 5. Plano de Validação

### 5.1 Validação Cruzada
- [ ] Usar **KFold** com k=5 ou k=10 (sem estratificação, pois é regressão)
- [ ] Métrica principal: **RMSPE** (implementar manualmente, pois o sklearn não tem essa métrica pronta)

### 5.2 Implementação da Métrica RMSPE

```python
import numpy as np

def rmspe(y_real, y_previsto):
    """Calcula o RMSPE (Root Mean Squared Percentage Error)"""
    return np.sqrt(np.mean(((y_real - y_previsto) / y_real) ** 2))
```

### 5.3 Acompanhamento de Resultados
- [ ] Manter uma tabela com todos os modelos testados e seus resultados:

| Modelo | Features | Hiperparâmetros | RMSPE (CV) | RMSPE (Kaggle) |
| :--- | :--- | :--- | :--- | :--- |
| Linear Regression | baseline | default | ? | - |
| Ridge | todas | alpha=? | ? | - |
| XGBoost | todas | lr=?, depth=? | ? | - |
| ... | ... | ... | ... | ... |

---

## 6. Plano de Submissão ao Kaggle

- [ ] Treinar o modelo final com **todos** os dados de treinamento (sem separar validação)
- [ ] Gerar previsões para o conjunto de teste
- [ ] Reverter a transformação log: `np.expm1(previsoes_log)`
- [ ] Salvar no formato exigido: `Id,preco`
- [ ] Submeter ao Kaggle e anotar o RMSPE real
- [ ] Iterar: ajustar features/modelo → resubmeter

---

## 7. Entregáveis Finais

- [ ] **Código-fonte** (script Python ou Jupyter Notebook) com todo o pipeline reproduzível
- [ ] **Relatório** descrevendo:
  - Pré-processamento realizado
  - Seleção de atributos e justificativa
  - Modelos experimentados e hiperparâmetros
  - Resultados intermediários e finais
  - Técnicas de validação utilizadas
- [ ] **E-mail para o professor** (heraldo@poli.ufrj.br) contendo:
  - ID do Kaggle
  - Código-fonte
  - Relatório (PDF ou integrado no notebook)

---



---

*Plano criado em 13/07/2026 — Será atualizado conforme avançarmos no projeto.*
