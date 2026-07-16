Após carregarmos os dados, confirmamos o numero de linhas , colunas , tipos e outras estatiscas básicas. Separamos então as features do target e verificamos possíveis erros no banco de dados como se havia algo vazio ou duplicado. Começamos então a investigar possíveis outliers, nesse projeto temos que analisar com um pouco mais de cuidado, porque um imóvel muito caro ainda pode ser plausível e o modelo deve ser capaz de ter essa noção dessa possibilidade.

Outliers:

Id / Decisão(se diferente de manter):

5910 / remover, claramente erro de digitação, um imóvel dificilmente seria vendido a 750 reais
2405 / remover, muito cima do preço mesmo tendo menos 100 metros quadrados de area util e sem area extra, mesmo sendo um bairro nobre não justifica um valor tão acima.
4568 / remover , preço incondizente com imóvel

5575 / remover, preco por metro quadrado ridiculo de barato, provavelmente esqueceu de uma virgula na area util
6606 / remover, muito metro quadrado com apenas 2 quartos e preço baixo

6654 / remover , 17450 metros quadrados de area extra é demais, e o preço não mostra isso.

6383 / asterisico, 30 vagas é um numero bem fora do padrao, mas de fato é um bom imóvel e não é barato, pode ser um caso real e raro.s

Após a identificação dos outliers, partimos para a criação de novas features que, podem tanto ajudar no treinamento de modelos mas também a identificar dados falhos mais escondidos, por exemplo a feature preco/m² depende de preço por isso nao pode treinar o modelo, porém é ótimo pra avaliarmos alguns dados incoerentes.

Features de Auditoria:

preco_m2 = preco / area_util
preco_por_quarto = preco / quartos
preco_por_vaga = preco / vagas , se vaga > 0.

A primeira a ser implementada foi preco_m2 com ela tivemos:

id/decisão

6004 /remover, preco por metro quadrado muito acima, certamente houve erro no numero de zeros do imovel

4316 / remover, preco por metro quadrado elevado demais para o bairro que é e pelo apartamento que é, sem nenhum diferencial que justifique o preço por metro quadrado.

Após isso buscaremos por preços de metro quadrado que distoem da mediana do bairro.Olharemos apenas para aqueles que possuem pelo menos 10 imoveis regisrados no bairro.

ID / Decisão(se diferente de manter)

3553 / remover, preço por metro quadrado muito abaixo da mediana do bairro, além de inconsistência física, 561 metros de área útil mas apenas 2 quartos e 1 suíte

4820 / remover, preço por metro quadrado muito abaixo da mediana do bairro.

3604 / remover, preço por metro quadrado muito abaixo da mediana do bairro.

2101 / remover, preço por metro quadrado muito abaixo da mediana do bairro.

 4220,6361,3937,3984,4075,2865,3998,3536,4708,3586,5540,3724,2705,4808,4280 / remover , preço por metro quadrado muito abaixo da mediana do bairro

2749 / preço muito acima
***Muitos desses dados devem ser falta de vírgula já que fariam sentido se por exemplo dividissimos a area por 10, é válido depois tentar ver se melhoramos mexendo nesses dados, além disso a tratativa foi conservadira tirando apenas outliers claros.

## Tratamento das variáveis categóricas

Depois da remoção dos casos considerados anômalos e da criação das features numéricas, tratamos as colunas categóricas para que elas pudessem ser utilizadas pelos modelos. Modelos como regressão linear, Ridge e Random Forest não trabalham diretamente com nomes como `Apartamento` ou `Boa Viagem`; esses valores precisam ser representados numericamente.

### Coluna `diferenciais`

A coluna `diferenciais` foi retirada de `df_modelo`. Ela contém descrições textuais de comodidades que já estão representadas separadamente pelas colunas binárias `churrasqueira`, `estacionamento`, `piscina`, `playground`, `quadra`, `s_festas`, `s_jogos`, `s_ginastica`, `sauna` e `vista_mar`. Manter as duas representações nesta primeira versão seria redundante. A retirada acontece apenas na tabela preparada em memória e não modifica o CSV original.

### Coluna `tipo`

Na análise inicial, a distribuição da coluna era muito desequilibrada: havia 4.501 apartamentos, 177 casas, 3 lofts e 2 quitinetes. Como `Loft` e `Quitinete` possuíam exemplos insuficientes para que o modelo aprendesse um comportamento confiável para cada categoria, eles foram agrupados na categoria `Outro`.

Depois do agrupamento, aplicamos one-hot encoding. A coluna de texto `tipo` foi substituída por três colunas binárias:

- `tipo_Apartamento`
- `tipo_Casa`
- `tipo_Outro`

Em cada linha, o valor é `1` na coluna correspondente ao tipo do imóvel e `0` nas demais. Dessa forma, não criamos uma ordem artificial entre os tipos de imóvel.

### Coluna `tipo_vendedor`

Como essa coluna possui apenas duas categorias, não era necessário criar duas colunas one-hot. Ela foi substituída por `vendedor_imobiliaria`, com a seguinte interpretação:

- `1`: o vendedor é uma imobiliária;
- `0`: o vendedor é uma pessoa física.

O nome da nova feature registra explicitamente o significado do valor `1`, evitando uma codificação numérica ambígua.

### Coluna `bairro`

Como primeira estratégia, aplicamos one-hot encoding a todos os 66 bairros. Cada bairro passou a ter uma coluna própria, como `bairro_Boa Viagem`, `bairro_Madalena` e `bairro_Torre`. Um imóvel localizado em Boa Viagem recebe `1` em `bairro_Boa Viagem` e `0` nas colunas dos outros bairros.

Essa estratégia foi escolhida por ser simples, transparente e por não usar o preço durante a transformação. Entretanto, alguns bairros possuem somente um ou dois imóveis. Colunas com tão poucos exemplos podem levar o modelo a memorizar casos específicos e causar sobreajuste. Depois de obtermos um resultado de referência, esta versão será comparada com alternativas como agrupar bairros raros em `Outros` ou utilizar target encoding com os cuidados necessários para evitar vazamento de dados.

O one-hot encoding foi executado com `pd.get_dummies`, usando `dtype=int`, para que as colunas geradas fossem armazenadas explicitamente como valores inteiros `0` e `1`. A transformação é realizada em `df_modelo` durante a execução do script; nenhuma dessas colunas é gravada no CSV original.

### Separação de identificador, entrada e alvo

Após todas as transformações, a coluna `Id` foi copiada para a variável `ids`. Ela será necessária para identificar cada imóvel e, futuramente, montar o arquivo de submissão, mas foi retirada da entrada do modelo porque seu número não descreve nenhuma característica do imóvel.

A separação final ficou definida da seguinte forma:

- `ids`: identificadores dos imóveis;
- `y`: coluna `preco`, que é o alvo que o modelo deverá prever;
- `x`: todas as features numéricas e categóricas transformadas, sem `Id` e sem `preco`.

### Verificação da transformação

Ao executar o script depois dessa etapa, foram confirmados os seguintes resultados:

- 4.654 imóveis restantes após as remoções;
- 91 colunas de entrada em `x`;
- 3 colunas one-hot referentes ao tipo de imóvel;
- 66 colunas one-hot referentes aos bairros;
- nenhuma coluna de texto restante em `x`;
- `vendedor_imobiliaria` contendo apenas os valores `0` e `1`;
- mesma quantidade de linhas em `x`, `y` e `ids`;
- ausência de `Id` e `preco` entre as features entregues ao modelo.

Com essas verificações, a primeira versão do conjunto de entrada ficou inteiramente numérica e pronta para iniciar a etapa de modelagem.
