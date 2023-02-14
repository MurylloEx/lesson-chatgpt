
# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Configuração do estilo dos gráficos
sns.set_style('whitegrid')

# Carregamento do dataset
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

df.describe()

df.hist()

df.head()

# Limpeza e preparação dos dados
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# Definição das variáveis independentes e dependentes
X = df.drop('Survived', axis=1)
y = df['Survived']

# Divisão dos dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Melhores parâmetros encontrados
print('Melhores parâmetros: ', grid_search.best_params_)

# Melhor desempenho encontrado
print('Melhor desempenho: ', grid_search.best_score_)

# Acurácia do modelo no conjunto de teste
print('Acurácia no conjunto de teste: ', grid_search.score(X_test, y_test))

# Curva ROC
probas = grid_search.predict_proba(X_test)
skplt.metrics.plot_roc(y_test, probas)

# Matriz de confusão
y_pred = grid_search.predict(X_test)
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)

# Importância das features
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': grid_search.best_estimator_.feature_importances_})
feature_importances = feature_importances.sort_values('importance', ascending=False)
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('Importância das features')
plt.show()

cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results = cv_results[['params', 'mean_test_score']]
cv_results = cv_results.sort_values('mean_test_score', ascending=False)
cv_results = cv_results.reset_index(drop=True)
cv_results = cv_results.head(10)
cv_results = pd.concat([cv_results['params'].apply(pd.Series), cv_results['mean_test_score']], axis=1)
cv_results = cv_results.rename(columns={'mean_test_score': 'score'})

sns.barplot(x='score', y='n_estimators', hue='max_features', data=cv_results)
plt.title('Scores de validação cruzada')
plt.show()
