# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplt
import streamlit as slt
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

slt.set_option('deprecation.showPyplotGlobalUse', False)

slt.title("Atividade com ChatGPT - Modelo Cross Validation e Grid Search")

slt.subheader("Configuração do estilo dos gráficos...")

# Configuração do estilo dos gráficos
sns.set_style('whitegrid')
# Define o tamanho dos gráficos
plt.rcParams['figure.figsize'] = (16,14)

slt.subheader("Carregamento do dataset...")

# Carregamento do dataset
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

slt.subheader("Exibindo algumas estatísticas do dataset")

slt.write(df.describe())

# Histogramas - diagramas de uma variável
numeric_cols = df.select_dtypes(include='number').columns.tolist()

df[numeric_cols].hist(bins=20, figsize=(10,10))
slt.pyplot()

slt.write(df.head())

slt.subheader("Criar gráfico de caixas (bloxplot)")

df.plot(kind='box', subplots=False, layout=(2,2))
slt.pyplot()

slt.subheader("Criar gráfico de realce dos outliers (lineplot)")

df.plot(kind='line', subplots=True, layout=(3,3))
slt.pyplot()

slt.subheader("Gráficos Multivariados (observar presença de agrupamentos diagonais indicando correlação)")

scatter_matrix(df)
slt.pyplot()

slt.subheader("Limpeza e preparação dos dados")

# Limpeza e preparação dos dados
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

slt.subheader("Definição das variáveis independentes e dependentes")

# Definição das variáveis independentes e dependentes
X = df.drop('Survived', axis=1)
y = df['Survived']

slt.subheader("Divisão dos dados em treinamento e teste")

# Divisão dos dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

slt.subheader("Definição do modelo e utilização do GridSearchCV")

# Definição do modelo a ser utilizado
rfc = RandomForestClassifier()

# Definição da grade de parâmetros a ser testada
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Definição da estratégia de validação cruzada
cv = 5

# Busca pelos melhores parâmetros com validação cruzada
grid_search = GridSearchCV(rfc, param_grid=param_grid, cv=cv, n_jobs=-1)
grid_search.fit(X_train, y_train)

slt.subheader("Exibindo os resultados (Melhores parâmetros, Melhor desempenho, Acurácia)")

# Melhores parâmetros encontrados
slt.write('Melhores parâmetros: ' + str(grid_search.best_params_))

# Melhor desempenho encontrado
slt.write('Melhor desempenho: ' + str(grid_search.best_score_))

# Acurácia do modelo no conjunto de teste
slt.write('Acurácia no conjunto de teste: ' + str(grid_search.score(X_test, y_test)))

slt.subheader("Curva ROC")

# Curva ROC
probas = grid_search.predict_proba(X_test)
skplt.metrics.plot_roc(y_test, probas)

slt.subheader("Matriz de confusão")

# Matriz de confusão
y_pred = grid_search.predict(X_test)
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)

slt.subheader("Importância das features")

# Importância das features
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': grid_search.best_estimator_.feature_importances_})
feature_importances = feature_importances.sort_values('importance', ascending=False)
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('Importância das features')
fig, ax = plt.subplots()
sns.barplot(data=feature_importances, ax=ax, kde=True)
slt.write(fig)

slt.subheader("Scores da validação cruzada")

cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results = cv_results[['params', 'mean_test_score']]
cv_results = cv_results.sort_values('mean_test_score', ascending=False)
cv_results = cv_results.reset_index(drop=True)
cv_results = cv_results.head(10)
cv_results = pd.concat([cv_results['params'].apply(pd.Series), cv_results['mean_test_score']], axis=1)
cv_results = cv_results.rename(columns={'mean_test_score': 'score'})

sns.barplot(x='score', y='n_estimators', hue='max_features', data=cv_results)
plt.title('Scores de validação cruzada')
fig, ax = plt.subplots()
sns.barplot(data=cv_results, ax=ax, kde=True)
slt.write(fig)
