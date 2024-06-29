# importation des librairies
import pandas as pd
import numpy as np
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, make_scorer, mean_squared_error
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
import xgboost as xgb

# importation de la base de données
prixlogement=pd.read_excel(r'c:\Users\ARMIDE Informatique\Downloads\donne_immobilier.xlsx')

# affichier les  cinq premières et dernières lignes de la base de données 
prixlogement.head()
prixlogement.tail()
prixlogement.info() #récupérer les informations sur la base de données 
msno.bar(prixlogement, color='red') #afficher le nombre d'observations de chaque variable
check_duplicated = prixlogement.duplicated().sum()
check_duplicated# afficher les doublons

#vérifier la présence des outliers
Q1 = prixlogement.quantile(0.25)
Q3 = prixlogement.quantile(0.75)
IQR = Q3 - Q1

# Afficher Q1, Q3, et IQR
print(f"Q1:\n{Q1}\n")
print(f"Q3:\n{Q3}\n")
print(f"IQR:\n{IQR}\n")

# Détecter les valeurs aberrantes
outliers = ((prixlogement < (Q1 - 1.5 * IQR)) | (prixlogement > (Q3 + 1.5 * IQR)))
print(outliers)

# Afficher les valeurs aberrantes
outliers_data = prixlogement[outliers.any(axis=1)]
print(outliers_data)

# Calculer les seuils
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Afficher les seuils
print(f"Seuil inférieur:\n{lower_bound}\n")
print(f"Seuil supérieur:\n{upper_bound}\n")

# Vérifier les valeurs aberrantes
outliers = ((prixlogement < lower_bound) | (prixlogement > upper_bound))

# Afficher le résultat
print(outliers)

# Afficher les valeurs aberrantes (s'il y en a)
outliers_data = prixlogement[outliers.any(axis=1)]
print(outliers_data)

# redéfinition de la base de données
prixlogement=prixlogement.drop(columns=['Unnamed: 0','date'])

# statistiques descriptives
statprixlogement=prixlogement.describe()
statprixlogement

# visualisation des données
corrprixlogement = prixlogement.corr()
sns.heatmap(corrprixlogement, annot= True, cmap='coolwarm') # visualisation graphique de la matrice de corrélation

sns.pairplot(prixlogement) # visualisation graphique des données

#visualisation de la relation entre la variable à prédire et les prédicteurs
sns.regplot(data=prixlogement, x='taux_volume_transaction', y='indice_prix_logement', color='black')
plt.title('relation entre l\'indice de prix de logement et le taux de volume de transaction')
plt.xlabel('taux de volume de transaction')
plt.ylabel("indice de prix de logement")
plt.show()

variables =['indice_prix_logement_lissé', 'volume_transaction', 'taux_occupation',
'taux_prix_logement', 'taux_volume_transaction','indice_prix_logement']

# Création de la figure et des axes
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Tracé des box plots sur chaque sous-graphe
for ax, var in zip(axes.flatten(), variables):
    sns.regplot(x='indice_prix_logement', y=var, data=prixlogement, ax=ax)
    ax.set_title(f'Relation entre {var} et l\'indice de prix de logement')

# Ajustement de l'espacement entre les sous-graphes
plt.tight_layout()
plt.show()

# normalisation et standardisation des données avant l'entraînement et l'évaluation de la performance prédictive

# Normalisation
min_max_scaler = MinMaxScaler()
data_normalized = prixlogement.copy()
data_normalized[variables] = min_max_scaler.fit_transform(prixlogement[variables])

# Afficher les données normalisées
print("Données normalisées :")
print(data_normalized.head())

# Standardisation
standard_scaler = StandardScaler()
data_standardized = prixlogement.copy()
data_standardized[variables] = standard_scaler.fit_transform(prixlogement[variables])

# Afficher les données standardisées
print("Données standardisées :")
print(data_standardized.head())

# Préparation des données normalisées pour l'entraînement et l'évaluation prédicitve
x = data_normalized.drop(columns=['indice_prix_logement']) # les prédicteurs
y= data_normalized['indice_prix_logement'] #variable à prédire

#division des données normalisées en sous-ensemble d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=808)

# Ajout des constants
x_train_const= sm.add_constant(x_train)
x_test_const= sm.add_constant(x_test)

# entraînement du modèle
model = sm.OLS(y_train, x_train_const).fit()
print(model.summary())

# prédiction
y_pred = model.predict(x_test_const)

# residus
residuals= y_test- y_pred

# test d'homoscédastiticité
lm_statistic, p_value_lm, f_statistic, p_value_f = het_breuschpagan(residuals, x_test_const)
# afficher les résultats
print("LM Statistic: ", lm_statistic)
print("LM p-value: ", p_value_lm)
print("F-statisitic", f_statistic)
print("F p-value", p_value_f)

# test d'autocorrélation des résuidus
durbin_watson_statistic=durbin_watson(residuals)
print("Statistic Durbin watson",durbin_watson_statistic)

# test de normalité des résidus
shapiro_statistic, p_value_shapiro = shapiro(residuals)

# évaluation de la performance prédictive
print("RMSE",mean_squared_error(y_test, y_pred))
print("MAE",mean_absolute_error(y_test, y_pred))

# autres modèles pour améliorer la robustesse de mon modèle

# Définir une fonction pour calculer le RMSE
# Définir une fonction pour calculer le RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Créer un scorer RMSE pour la validation croisée
rmse_scorer = make_scorer(rmse, greater_is_better=False)

# Modèles à tester
models = {
    'Linear Regression': sm.OLS(y_train, x_train_const),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=808)
}

# Evaluation des modèles
# Evaluation des modèles
print("Évaluation des modèles sur données normalisées:")
for name, model in models.items():
    if name == 'Linear Regression':
        results = model.fit().predict(x_test_const)
        score = rmse(y_test, results)
    else:
        model.fit(x_train_const, y_train)
        y_pred = model.predict(x_test_const)
        score = rmse(y_test, y_pred)
    print(f"{name}: RMSE = {score:.4f}")


# Définir le modèle de forêt aléatoire
model_rf = RandomForestRegressor(random_state=808)

# Définir les hyperparamètres à optimiser
param_grid = {
    'n_estimators': [50, 100, 200],  # Nombre d'arbres dans la forêt
    'max_depth': [None, 10, 20, 30],  # Profondeur maximale des arbres
    'min_samples_split': [2, 5, 10]  # Nombre minimum d'échantillons requis pour scinder un nœud
}

# Utiliser GridSearchCV pour trouver les meilleurs hyperparamètres
grid_search = GridSearchCV(estimator=model_rf, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)
grid_search.fit(x_train_const, y_train)

# Afficher les meilleurs hyperparamètres trouvés
print("Meilleurs hyperparamètres trouvés :")
print(grid_search.best_params_)

# Obtenez le meilleur modèle
best_model_rf = grid_search.best_estimator_
print(best_model_rf)

# Définir la validation croisée (par exemple, K-Fold avec K=5)
kf = KFold(n_splits=5, shuffle=True, random_state=808)

# Liste pour stocker les résultats de la validation croisée
cv_results = []

# Effectuer la validation croisée
for train_index, val_index in kf.split(x_train_const):
    x_train_fold, x_val_fold = x_train_const.iloc[train_index], x_train_const.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    # Entraîner le meilleur modèle sur le pli actuel
    best_model_rf.fit(x_train_fold, y_train_fold)
    
    # Faire des prédictions sur l'ensemble de validation
    y_pred_fold = best_model_rf.predict(x_val_fold)
    
    # Calculer l'erreur quadratique moyenne sur l'ensemble de validation
    fold_rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
    cv_results.append(fold_rmse)

# Calculer la moyenne et l'écart-type des résultats de validation croisée
mean_cv_rmse = np.mean(cv_results)
std_cv_rmse = np.std(cv_results)

print(f"Validation croisée RMSE moyenne : {mean_cv_rmse:.4f} +/- {std_cv_rmse:.4f}")

# Import des bibliothèques nécessaires
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold

# Définir les hyperparamètres à rechercher
params = {
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 3, 5],
    'gamma': [0.0, 0.1, 0.2],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Initialiser le modèle XGBoost
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=808)

# Utiliser Grid Search avec validation croisée pour trouver les meilleurs hyperparamètres
grid_search = GridSearchCV(estimator=xgb_model, param_grid=params, scoring='neg_mean_squared_error', cv=KFold(n_splits=5, shuffle=True, random_state=808))
grid_search.fit(x_train_const, y_train)

# Afficher les meilleurs hyperparamètres trouvés
print("Meilleurs hyperparamètres trouvés :", grid_search.best_params_)

# Entraîner le modèle avec les meilleurs hyperparamètres trouvés
best_xgb_model = grid_search.best_estimator_
best_xgb_model.fit(x_train_const, y_train)

# Prédire sur l'ensemble de test
y_pred = best_xgb_model.predict(x_test_const)

# Calculer et afficher le RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE avec XGBoost : {rmse:.4f}")
