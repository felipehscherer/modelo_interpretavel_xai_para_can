import time

import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, ConfusionMatrixDisplay, \
    confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Instanciar o codificador
encoder = LabelEncoder()

# Carregar os datasets
# df1 é o dataset sem ataque e o df2 com ataque
df = pd.read_parquet('repository/dump6.parquet')

# Combinar os datasets

# Codificar colunas do tipo object
df['112_CUR_GR'] = encoder.fit_transform(df['112_CUR_GR'])

# Dividir os dados novamente após a codificação
X = df.drop('label', axis=1)
y = df['label']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

before = time.time()
# Configurar e treinar o modelo XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)


# Avaliar o modelo
f1score = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
acuracia = accuracy_score(y_test, y_pred)


explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_train)
# Plotar os valores SHAP para a importância das features
shap.summary_plot(shap_values, X_train)


after = time.time()
time = (after - before)


print(f1score)
print(recall)
print(precision)
print(acuracia)
print(time)


