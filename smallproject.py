# Importation des bibliothèques
import pandas as pd
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge,LinearRegression

# Importation de la base de données
canada = pd.read_excel(r"d:\Projet Informatique\Bases de données\Base Crowdfunding.xlsx")
canada.tail() #Affichage des 05 dernières lignes
canadl = canada.select_dtypes(exclude=['object'])

# Aanalyses exploratoires des données
# Détection et supression des données manquantes
missing_data = canadl[canadl.duplicated()]
if not missing_data.empty:
    print('La base de données contient des données manquantes')
    msno.bar(canadl)
    canadl = canadl.drop_duplicates()
else:
    print("La base de données est sans danger")
    canadl

# Détection des valeurs aberrantes
Q1 = canadl.quantile(0.25)
Q3 = canadl.quantile(0.75)
IQR = Q3 - Q1
outlier = ((canadl < (Q1 - 1.5 * IQR)) | (canadl > (Q3 + 1.5 * IQR)))
valeurs_aberrantes = canadl[outlier.any(axis=1)] 

if not valeurs_aberrantes.empty:
    print('Datasets contains outliers')
    valeurs_aberrantes
else:
    print('Dataset is cleaned, no danger', canadl)

canadl = canadl.drop(valeurs_aberrantes.index) #Sup^pression des données abérrantes

# Statistique descriptive
statistic_canada = canadl.describe()
statistic_canada #résultat de la statistique

# Matrice de corrélation
corr_canada = canadl.corr()
sns.heatmap(corr_canada, annot=True, cmap='magma')

# Visualisation des variables 
sns.pairplot(canadl, palette="cividis")

# Visualisation des relations entre la cible et les prédicteurs
fig, ax = plt.subplots(1,2, figsize =(14,6))
sns.regplot(data=canadl, x='Revenus nets  (en milliers d\'€)', y= 'Chiffres d\'affaires (en milliers d\'€)', color='purple', ax=ax[0] )
ax[0].set(
    title=('Relation en les revenus nets et le chiffre d\'affaire'),
    xlabel=('Revenus nets'),
    ylabel=('Chiffred d\'affaire')
)
sns.regplot(data=canadl, x='Montant de financement participatif  (en milliers d\'€)', y= 'Chiffres d\'affaires (en milliers d\'€)', color='orange', ax=ax[1] )
ax[1].set(
    title=('Relation en les revenus nets et le chiffre d\'affaire'),
    xlabel=('Montant de financement participatif'),
    ylabel=('Chiffred d\'affaire')
)

# Normalisation des données
scaler =MinMaxScaler()
variables = ['Chiffres d\'affaires (en milliers d\'€)','Montant de financement participatif  (en milliers d\'€)','Revenus nets  (en milliers d\'€)']
canadl[variables]= scaler.fit_transform(canadl[variables])
canadl #Base de données normalisée
canadl =canadl.drop(columns=['ID_des entreprises']) #suppression de la variable id des entreprises

# Préparation des données
x = canadl.drop(columns=['Chiffres d\'affaires (en milliers d\'€)']) #les prédicteurs
y = canadl['Chiffres d\'affaires (en milliers d\'€)'] #la cible

# Division des données en ensemble d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=808)

# Ajout des constants
x_train_const=sm.add_constant(x_train)
x_test_const=sm.add_constant(x_test)

# Entraînement du modèle 
model=sm.OLS(y_train,x_train_const).fit()
print(model.summary())

# Prédiction
y_pred = model.predict(x_test_const)

# Evaluation de la capacité prédictive
print('MAE:',mean_absolute_error(y_pred,y_test))
print('MAPE:', mean_absolute_percentage_error(y_pred, y_test))

# Tester le modèle de la forêt aléatoire 
# Initialisation du modèle
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Entraînement du modèle
rf_model.fit(x_train_const, y_train)

# Prédictions
y_pred_rf = rf_model.predict(x_test_const)

# Évaluation
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf)

print(f"Random Forest MAE: {mae_rf}")
print(f"Random Forest MAPE: {mape_rf}")

# Méthode régularisée (Lasso et Ridge)
# Initialisation des modèles
ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=0.1)

# Entraînement des modèles
ridge_model.fit(x_train_const, y_train)
lasso_model.fit(x_train_const, y_train)

# Prédictions
y_pred_ridge = ridge_model.predict(x_test_const)
y_pred_lasso = lasso_model.predict(x_test_const)

# Évaluation
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
mape_ridge = mean_absolute_percentage_error(y_test, y_pred_ridge)

mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
mape_lasso = mean_absolute_percentage_error(y_test, y_pred_lasso)

print(f"Ridge MAE: {mae_ridge}")
print(f"Ridge MAPE: {mape_ridge}")
print(f"Lasso MAE: {mae_lasso}")
print(f"Lasso MAPE: {mape_lasso}")

# Améliortion de la capacité prédictive du modèle "Forêt aléatoire"
# Définir les paramètres à tester
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialisation du modèle
rf_model = RandomForestRegressor(random_state=42)

# Grid Search
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(x_train_const, y_train)

# Meilleurs paramètres
print(f"Best parameters: {grid_search.best_params_}")

# Prédictions avec le meilleur modèle
best_rf_model = grid_search.best_estimator_
y_pred_best_rf = best_rf_model.predict(x_test_const)

# Évaluation
mae_best_rf = mean_absolute_error(y_test, y_pred_best_rf)
mape_best_rf = mean_absolute_percentage_error(y_test, y_pred_best_rf)

print(f"Optimized Random Forest MAE: {mae_best_rf}")

# Feature engineering
# Exemple de création de nouvelles caractéristiques
canadl['new_var'] = canadl['Revenus nets  (en milliers d\'€)'] * canadl['Montant de financement participatif  (en milliers d\'€)']

# Réentraînement du modèle avec les nouvelles caractéristiques
x = canadl.drop(columns=['Chiffres d\'affaires (en milliers d\'€)'])
y = canadl['Chiffres d\'affaires (en milliers d\'€)']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=808)
x_train_const = sm.add_constant(x_train)
x_test_const = sm.add_constant(x_test)

best_rf_model.fit(x_train_const, y_train)
y_pred_best_rf = best_rf_model.predict(x_test_const)

mae_best_rf = mean_absolute_error(y_test, y_pred_best_rf)
mape_best_rf = mean_absolute_percentage_error(y_test, y_pred_best_rf)

print(f"Optimized Random Forest with New Features MAE: {mae_best_rf}")

# Analyse des résidus
# Calcul des résidus
residuals = y_test - y_pred_best_rf

# Visualisation des résidus
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_best_rf, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Prédictions')
plt.ylabel('Résidus')
plt.title('Analyse des Résidus')
plt.show()

# Pas d'optimisation des paramètres mais plutôt une validation croisée avec la régression linéaire
# Initialisation du modèle
lr_model = LinearRegression()

# Validation croisée
cv_scores = cross_val_score(lr_model, x_train_const, y_train, cv=5, scoring='neg_mean_absolute_error')

# Affichage des scores
print(f"Cross-validated MAE: {-cv_scores.mean()}")

# OPTIMISATION DES HYPERPARAMETRES ET VALIDATION CROISEE AVEC LES METHODES REGULARISEES
# Définir les paramètres à tester
param_grid_lasso = {
    'alpha': [0.01, 0.1, 1, 10, 100]
}

# Initialisation du modèle
lasso_model = Lasso(random_state=42)

# Grid Search avec validation croisée
grid_search_lasso = GridSearchCV(estimator=lasso_model, param_grid=param_grid_lasso, cv=5, n_jobs=-1, verbose=2)
grid_search_lasso.fit(x_train_const, y_train)

# Meilleurs paramètres
print(f"Best parameters for Lasso: {grid_search_lasso.best_params_}")

# Prédictions avec le meilleur modèle
best_lasso_model = grid_search_lasso.best_estimator_
y_pred_best_lasso = best_lasso_model.predict(x_test_const)

# Évaluation
mae_best_lasso = mean_absolute_error(y_test, y_pred_best_lasso)
print(f"Optimized Lasso MAE: {mae_best_lasso}")

from sklearn.linear_model import Ridge

# Définir les paramètres à tester
param_grid_ridge = {
    'alpha': [0.01, 0.1, 1, 10, 100]
}

# Initialisation du modèle
ridge_model = Ridge(random_state=42)

# Grid Search avec validation croisée
grid_search_ridge = GridSearchCV(estimator=ridge_model, param_grid=param_grid_ridge, cv=5, n_jobs=-1, verbose=2)
grid_search_ridge.fit(x_train_const, y_train)

# Meilleurs paramètres
print(f"Best parameters for Ridge: {grid_search_ridge.best_params_}")

# Prédictions avec le meilleur modèle
best_ridge_model = grid_search_ridge.best_estimator_
y_pred_best_ridge = best_ridge_model.predict(x_test_const)

# Évaluation
mae_best_ridge = mean_absolute_error(y_test, y_pred_best_ridge)
print(f"Optimized Ridge MAE: {mae_best_ridge}")
