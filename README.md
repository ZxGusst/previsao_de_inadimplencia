Acessar aplicação:
https://previsaodeinadimplencia-44hpw9znrgzhfnef7xgxxw.streamlit.app

# Previsão de Inadimplência de Crédito

## Grupo 8

- **Cayky Emilio Vieira Neves**
- **Gustavo dos Santos Alves**
- **Matheus Neiva Soares**
- **Thiago Pereira Dantas**

---

## Introdução

Neste projeto, atuamos como analistas de dados de uma fintech com o objetivo de construir um modelo preditivo capaz de identificar clientes com maior risco de inadimplência em empréstimos. Utilizamos um conjunto de dados contendo variáveis como idade, renda, saldo bancário, histórico de pagamento e uma coluna-alvo indicando se houve inadimplência. O projeto envolve etapas de limpeza, análise exploratória (EDA), pré-processamento, modelagem com algoritmos de classificação e avaliação com métricas de desempenho.

---

## 1. Leitura e Tratamento de Dados

- **Leitura do arquivo** `credit_data.csv` **com a biblioteca Pandas.**
- **Verificação e tratamento de valores ausentes** (preenchidos com a média).
- **Exclusão da coluna ID** (se existente).
- **Análise estatística básica** com `.describe()`.

---

## 2. Análise Exploratória de Dados (EDA)

- A variável-alvo `default.payment.next.month` apresentou desequilíbrio de classes (mais clientes adimplentes do que inadimplentes).
- A matriz de correlação mostrou baixa correlação entre muitas variáveis, o que indica relativa independência.
- Algumas variáveis numéricas apresentaram possíveis outliers, especialmente nos valores de saldo e pagamento.
- Gráficos utilizados: histogramas, boxplots e heatmap de correlação.

---

## 3. Pré-processamento

- **Codificação de variáveis categóricas** (se necessário).
- **Normalização dos dados** com `StandardScaler`.
- **Separação entre dados de treino e teste**: 70% treino / 30% teste.
- **Definição de variáveis independentes** (X) e **dependente** (y).

---

## 4. Modelos Treinados

### 4.1 Regressão Logística

- **Modelo simples e interpretável.**
- **Resultados:**
  - Acurácia: 0.8088
  - Precisão: 0.6786
  - Recall: 0.2316
  - F1-score: 0.3454
  - AUC-ROC: 0.6005

*Matriz de Confusão: incluída no notebook*

### 4.2 Random Forest (com GridSearchCV)

- **Modelo mais robusto e com melhor desempenho.**
- **Hiperparâmetros ajustados com GridSearchCV:**
  - `n_estimators`: 100
  - `max_depth`: None
  - `min_samples_split`: 5

- **Resultados:**
  - Acurácia: 0.8153
  - Recall: 0.8153

---

## 5. Conclusões e Recomendações

- O modelo **Random Forest** apresentou o melhor desempenho, especialmente no recall, que é uma métrica importante em cenários de inadimplência (prioridade: evitar falsos negativos).
- O sistema pode ser utilizado como ferramenta de apoio à concessão de crédito, automatizando a avaliação de risco.

**Limitações:**
- O dataset é limitado ao histórico fornecido, sem dados externos (como CPF ou histórico bancário mais abrangente).
- Pode haver viés nos dados ou falta de atualizações temporais.

---

## 6. Tecnologias Utilizadas

- **Linguagem:** Python 3.8+
- **Ambiente:** Jupyter Notebook
- **Bibliotecas:**
  - `pandas`, `numpy` para manipulação de dados
