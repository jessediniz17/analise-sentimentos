# 🧠 Análise de Sentimentos com Machine Learning

> Modelo de classificação de sentimentos aplicado a avaliações de produtos de e-commerce, desenvolvido com Python e Scikit-learn.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

---

## 📌 Sobre o projeto

Uma empresa de e-commerce recebe diariamente um grande volume de avaliações de seus clientes. Quando essa análise é realizada manualmente, o processo se torna lento, custoso e pouco escalável.

Este projeto propõe uma solução de **Machine Learning para classificação automática de sentimentos**, capaz de categorizar avaliações de produtos como:

- 🟢 **Positivo**
- 🔴 **Negativo**

O projeto foi construído pensando não apenas no treinamento de um modelo, mas em um fluxo completo de desenvolvimento de uma solução preditiva, passando por **análise exploratória, preparação dos dados, engenharia de atributos, otimização de hiperparâmetros, avaliação e preparação para utilização em produção**.

---

## 🎯 Problema de negócio

A classificação automática dos reviews pode apoiar diferentes áreas da empresa:

| Necessidade | Possível aplicação |
|---|---|
| Analisar grandes volumes de reviews | Automatização da classificação |
| Identificar problemas rapidamente | Priorização de avaliações negativas |
| Apoiar equipes de produto | Identificação de oportunidades de melhoria |
| Melhorar atendimento | Encaminhamento de feedbacks negativos |
| Reduzir trabalho manual | Processamento automatizado |

A ideia central é transformar **texto não estruturado em informação acionável para o negócio**.

---

## 🏗️ Pipeline da solução

O projeto segue um fluxo estruturado de Machine Learning:

```text
Dataset
   │
   ▼
Análise exploratória
   │
   ▼
Limpeza e normalização dos textos
   │
   ▼
Engenharia de atributos
   │
   ▼
Divisão treino / teste
   │
   ▼
Pipeline Scikit-learn
   │
   ├── TF-IDF
   ├── StandardScaler
   └── Logistic Regression
   │
   ▼
GridSearchCV + Cross-Validation
   │
   ▼
Melhor modelo
   │
   ▼
Avaliação
   ├── Accuracy
   ├── Precision
   ├── Recall
   ├── F1-Score
   └── Confusion Matrix
   │
   ▼
Persistência com Joblib
   │
   ▼
Predição de novos reviews
```

---

## 📊 Dataset

O notebook trabalha com um dataset contendo **500 registros e 3 colunas**.

O conjunto de dados é inicialmente carregado utilizando Pandas e passa por uma etapa de inspeção para compreender:

- estrutura dos dados;
- quantidade de registros;
- valores ausentes;
- distribuição das classes;
- exemplos dos reviews.

A análise exploratória também verifica a distribuição entre sentimentos positivos e negativos antes do treinamento.

---

## 🧹 Pré-processamento de texto

Uma etapa importante do projeto foi a criação de uma função própria para normalização dos reviews.

A função `limpar_texto()` realiza:

1. Conversão para letras minúsculas;
2. Normalização Unicode;
3. Remoção de acentos e cedilha;
4. Remoção de pontuação;
5. Remoção de números e caracteres especiais;
6. Normalização de espaços.

Exemplo conceitual:

```text
"Amei o produto! Ótima qualidade."
                ↓
"amei o produto otima qualidade"
```

Essa etapa reduz variações desnecessárias na representação textual e prepara os dados para a vetorização.

---

## 🔢 Engenharia de atributos

As avaliações textuais precisam ser transformadas em uma representação numérica para que possam ser utilizadas pelo algoritmo.

Além da criação do texto normalizado, as classes também são transformadas em valores numéricos:

```text
positivo → 1
negativo → 0
```

A representação textual é posteriormente realizada através do **TF-IDF (Term Frequency–Inverse Document Frequency)**.

Essa escolha permite representar a importância relativa das palavras dentro dos documentos sem depender de modelos de linguagem complexos.

---

## 🤖 Modelo escolhido

O classificador utilizado foi a **Regressão Logística**.

A escolha é adequada para este tipo de problema porque permite construir um classificador eficiente para dados textuais representados como vetores esparsos.

O modelo também apresenta uma vantagem importante para um projeto de portfólio: sua abordagem é relativamente simples de interpretar, treinar e colocar em produção.

A arquitetura final utiliza:

```text
TF-IDF
   ↓
StandardScaler
   ↓
Logistic Regression
```

---

## 🔗 Por que utilizar um Pipeline?

Uma das decisões técnicas mais importantes do projeto foi encapsular o pré-processamento e o modelo dentro de um único `Pipeline` do Scikit-learn.

```python
Pipeline([
    ('tfidf', TfidfVectorizer(...)),
    ('scaler', StandardScaler(with_mean=False)),
    ('logreg', LogisticRegression(...))
])
```

Isso evita que as etapas de transformação sejam executadas manualmente e garante que **o mesmo fluxo aplicado durante o treinamento seja utilizado posteriormente durante as previsões**.

Outro detalhe importante é:

```python
StandardScaler(with_mean=False)
```

O `TF-IDF` produz uma matriz esparsa. Desabilitar o cálculo da média evita transformar essa estrutura em uma matriz densa, preservando uma representação mais eficiente em termos computacionais.

Essa é uma escolha técnica relevante porque demonstra preocupação não apenas com o funcionamento do modelo, mas também com a **estrutura dos dados e eficiência do processamento**.

---

## ⚙️ Otimização de hiperparâmetros

Em vez de utilizar os hiperparâmetros padrão, o projeto utiliza `GridSearchCV` para procurar automaticamente uma configuração mais adequada.

Foram testadas combinações envolvendo:

- `max_features` do TF-IDF;
- `ngram_range`;
- `C` da Regressão Logística;
- `penalty`;
- `max_iter`.

O processo utilizou:

```text
5-fold Cross-Validation
```

e avaliou:

```text
72 combinações
×
5 folds
=
360 ajustes
```

A métrica utilizada para seleção foi **accuracy**.

### 🏆 Melhor configuração encontrada

```python
{
    'logreg__C': 0.1,
    'logreg__max_iter': 5000,
    'logreg__penalty': 'l1',
    'tfidf__max_features': 500,
    'tfidf__ngram_range': (1, 1)
}
```

Um ponto interessante é que o processo selecionou uma representação relativamente enxuta:

- 500 features;
- apenas unigramas;
- regularização L1;
- `C = 0.1`.

Isso mostra que o melhor resultado encontrado não necessariamente exige uma representação textual extremamente complexa.

---

## 📈 Avaliação

Após o treinamento, o modelo foi avaliado utilizando um conjunto de teste separado durante o processo inicial.

### Resultado

**Accuracy: 81,15%**

| Classe | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Negativo | 0,80 | 0,81 | 0,80 |
| Positivo | 0,83 | 0,81 | 0,82 |
| **Média** | **0,81** | **0,81** | **0,81** |

O conjunto de teste contém **122 reviews**.

A proximidade entre precision, recall e F1-score nas duas classes indica um comportamento relativamente equilibrado entre as classificações positivas e negativas.

Além das métricas numéricas, o projeto utiliza uma **matriz de confusão**, permitindo visualizar os acertos e erros de classificação por classe.

---

## 💾 Persistência do modelo

Depois da avaliação, o pipeline treinado é salvo utilizando Joblib:

```python
joblib.dump(melhor_modelo, 'modelo_sentimento_v1.joblib')
```

Essa decisão permite separar o processo de **treinamento** do processo de **inferência**.

O modelo pode posteriormente ser carregado:

```python
modelo_deploy = joblib.load('modelo_sentimento_v1.joblib')
```

Assim, não é necessário treinar novamente o modelo toda vez que uma nova avaliação precisar ser classificada.

---

## 🚀 Simulação de Deploy

O notebook também demonstra como o modelo poderia ser utilizado em uma aplicação real.

Foi criada uma função de inferência:

```python
def prever_sentimento(reviews):
    ...
```

Ela recebe uma lista de avaliações e retorna automaticamente a classificação prevista.

Exemplos utilizados no projeto:

```text
"A bateria do celular não dura nada, péssima compra."
→ Negativo

"Chegou antes do prazo e o produto é de ótima qualidade!"
→ Positivo

"O serviço de atendimento foi rápido e eficiente."
→ Negativo

"Não recomendo, veio faltando peças e a cor estava errada."
→ Negativo
```

O ponto importante aqui é que o consumidor da função não precisa conhecer as etapas internas do modelo.

O pipeline cuida automaticamente de:

```text
Texto
 ↓
TF-IDF
 ↓
Scaler
 ↓
Classificador
 ↓
Sentimento
```

Isso aproxima o notebook de uma estrutura que poderia posteriormente ser integrada a uma **API, aplicação web ou pipeline de processamento de reviews**.

---

## 🧰 Tecnologias utilizadas

### Linguagem

- **Python**

### Análise e manipulação de dados

- Pandas
- NumPy

### Visualização

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- TF-IDF
- Logistic Regression
- GridSearchCV
- Cross-Validation
- Pipeline
- Classification Report
- Confusion Matrix

### Persistência

- Joblib

### Ambiente

- Jupyter Notebook

---

## 📁 Estrutura sugerida

```text
analise-sentimentos/
│
├── analise_sentimentos.ipynb
├── dataset.csv
├── modelo_sentimento_v1.joblib
└── README.md
```

> O arquivo `.joblib` é gerado após o treinamento do modelo.

---

## 🧠 Principais decisões técnicas

### 1. TF-IDF em vez de uma abordagem mais complexa

O projeto utiliza TF-IDF por ser uma solução eficiente e adequada para um problema de classificação binária com dataset relativamente pequeno.

Isso também mantém o pipeline simples, rápido e fácil de reproduzir.

### 2. Regressão Logística

A Regressão Logística oferece uma solução robusta para classificação textual quando combinada com representações como TF-IDF.

A escolha também favorece simplicidade e interpretabilidade.

### 3. Pipeline

O uso de `Pipeline` reduz o risco de inconsistências entre treinamento e inferência e torna o modelo mais fácil de reutilizar.

### 4. Cross-Validation

O `GridSearchCV` com 5 folds reduz a dependência de uma única divisão dos dados durante a seleção dos hiperparâmetros.

### 5. Separação entre treinamento e inferência

A persistência através do Joblib permite que o modelo treinado seja reutilizado posteriormente sem repetir todo o processo de treinamento.

---

## 💡 Pontos fortes do projeto

Este projeto demonstra conhecimentos importantes para uma posição de **Data Scientist / Machine Learning Engineer**:

- Desenvolvimento de solução orientada a problema de negócio;
- Análise exploratória de dados;
- Tratamento de dados textuais;
- Feature engineering;
- Classificação supervisionada;
- Construção de pipelines de Machine Learning;
- Otimização de hiperparâmetros;
- Validação cruzada;
- Avaliação através de múltiplas métricas;
- Persistência de modelos;
- Separação entre treinamento e inferência;
- Estruturação de uma solução com possibilidade de evolução para produção.

Mais do que simplesmente treinar um algoritmo, o projeto demonstra a construção de um **ciclo completo de Machine Learning**.

---

## 🔮 Próximos passos

O projeto pode ser evoluído para uma solução mais próxima de produção através de:

- [ ] Criar uma API REST com FastAPI;
- [ ] Criar uma interface interativa com Streamlit;
- [ ] Adicionar Docker;
- [ ] Criar testes automatizados;
- [ ] Implementar logging;
- [ ] Criar monitoramento de performance;
- [ ] Avaliar outros modelos de classificação;
- [ ] Comparar TF-IDF com embeddings;
- [ ] Experimentar modelos baseados em Transformers;
- [ ] Criar uma etapa de retreinamento periódico;
- [ ] Implementar uma arquitetura de inferência em produção.

---

## 📚 Objetivo de aprendizado

Este projeto faz parte da construção de uma base prática em **Data Science, Machine Learning e Engenharia de IA**, com foco em transformar conceitos estatísticos e algoritmos de aprendizado de máquina em soluções aplicáveis a problemas reais.

A proposta não é buscar apenas a maior métrica possível, mas compreender todo o processo necessário para transformar **dados brutos em uma solução preditiva reutilizável**.

---

## 👨‍💻 Autor

**Jessé Diniz**

Profissional de tecnologia com foco em **Data Science, Machine Learning e AI Engineering**, desenvolvendo soluções que combinam Python, dados, automação e Inteligência Artificial.

---

⭐ Se este projeto foi útil ou interessante, considere deixar uma estrela no repositório.
