# README

## Introdução

Este projeto contém um script Python para plotar gráficos SHAP de um dataset CAN (Controller Area Network). Os gráficos SHAP são usados para explicar as previsões de um modelo de machine learning, mostrando a importância de cada característica na tomada de decisão do modelo.

## Descrição do Script

O script realiza os seguintes passos principais:

1. **Importação das Bibliotecas Necessárias**:
    - Importa bibliotecas essenciais como `numpy`, `pandas`, `shap`, `XGBClassifier` do `xgboost`, funções de `train_test_split` do `sklearn.model_selection`, métricas de `sklearn.metrics` e `matplotlib.pyplot`.

2. **Carregamento do Dataset**:
    - Carrega um dataset CAN no formato CSV utilizando `pandas`.

3. **Pré-processamento dos Dados**:
    - Separa o dataset em características (`features`) e o alvo (`target`).
    - Divide o dataset em conjuntos de treinamento e teste utilizando uma proporção de 80/20.

4. **Treinamento do Modelo**:
    - Treina um modelo `XGBClassifier` com os dados de treinamento.

5. **Cálculo dos Valores SHAP**:
    - Utiliza `shap.TreeExplainer` para calcular os valores SHAP, que explicam a importância das características nas previsões do modelo.

6. **Plotagem dos Gráficos SHAP**:
    - Gera gráficos SHAP para visualizar a importância das características. O primeiro gráfico é um gráfico de barras que mostra a importância média das características. O segundo é um gráfico de resumo que mostra a dispersão dos valores SHAP para cada característica.

## Configuração

Para configurar o ambiente e executar o script, siga os passos abaixo. Certifique-se de ter o Python 3.12 instalado.

### Requisitos

Instale as dependências necessárias utilizando `pip`:

```sh
pip install numpy
pip install pandas
pip install shap
pip install xgboost
pip install scikit-learn 
```

Para manipular os arquivos .parquet do dataset sugiro:

```sh
pip install fastparquet

