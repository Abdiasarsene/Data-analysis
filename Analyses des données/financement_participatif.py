# importation de la base de données
import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error,make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold

# importation de la base de données
unknowndata= pd.read_excel(r'c:\Users\ARMIDE Informatique\Desktop\Projet Informatique\Base Crowdfunding_2.xlsx')
unknowndata #affichage des données du jeu de données

"PREPARATION DE LA BASE DE DONNEES POUR UNE PREDICTION"

infodata=unknowndata.info()#résumé sur le dataset
infodata
unknowndata = unknowndata.drop(columns=['ID_des entreprises','Type de financement  ']) #suppression de certaines variables inutiles pour notre modelisation
msno.bar(unknowndata, color='orange')#afficher les valeurs manquantes
duplicatedata= unknowndata.duplicated().sum()#afficher les doublons

# détection des outliers avec la méthode iqr
Q1 = unknowndata.quantile(0.25)
Q3 = unknowndata.quantile(0.75)
IQR = Q3-Q1
outliers=((unknowndata<(Q1-1.5*IQR))|(unknowndata >(Q3+1.5*IQR)))
valeurs_aberrantes=unknowndata[outliers.any(axis=1)]

# affichage des résultats
print("les doublons sont : ",duplicatedata)
print("les valeurs aberrantes sont : ")
valeurs_aberrantes

# suppression des lignes contenant des outliers
unknowndata= unknowndata.drop(valeurs_aberrantes.index)

"Statistique descriptive et graphique de la matrice de corrélation"
stat=unknowndata.describe()
stat #afficher la statistique
corree= unknowndata.corr()
sns.heatmap(corree, annot=True, cmap='coolwarm')#afficher la matrice de corrélation sous forme graphique

"visualisation des données"
sns.pairplot(unknowndata) #visualisation graphique des données

# relation entre la variable à prédire et les prédicteurs
fig, ax=plt.subplots(1,2, figsize=(14,6))
sns.regplot(data=unknowndata, x='Chiffres d\'affaires (en milliers d\'€)', y= 'Montant de financement participatif  (en milliers d\'€)', color='green', ax=ax[0])
ax[0].set(
    title=('chiffre d\'affaire vs le montant de financement participatif'),
    xlabel=('chiffre d\'affaire'),
    ylabel=('montant de financement participatif')
)

sns.regplot(data=unknowndata, x='Revenus nets  (en milliers d\'€)', y= 'Montant de financement participatif  (en milliers d\'€)', color='purple', ax=ax[1])
ax[1].set(
    title=('Revenus nets  vs le montant de financement participatif'),
    xlabel=('Revenus nets'),
    ylabel=('montant de financement participatif')
)


# Normalisation
variables =['Chiffres d\'affaires (en milliers d\'€)','Montant de financement participatif  (en milliers d\'€)', 'Revenus nets  (en milliers d\'€)']
min_max_scaler = MinMaxScaler()
data_normalized = unknowndata.copy()
data_normalized[variables] = min_max_scaler.fit_transform(unknowndata[variables])

# Préparation des données normalisées pour l'entraînement et l'évaluation prédicitve
x = data_normalized.drop(columns=['Montant de financement participatif  (en milliers d\'€)']) # les prédicteurs
y= data_normalized['Montant de financement participatif  (en milliers d\'€)'] #variable à prédire

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

"PREMIERE EVALUATION DE LA PERFORMANCE PREDICTIVE DE NOTRE PROJET"


# évaluation de la performance prédictive
print("RMSE",mean_squared_error(y_test, y_pred))
print("MAPE",mean_absolute_percentage_error(y_test, y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


"AUTRES MODELES TESTER POUR TROUVER LE MODELE QUI EVALUE BIEN LA PERFORMANCE DU MODELE : Dans ce cas, nous avons utilisé autres modèles comme les méthodes régularisées (les regressions ridge et lasso) et l'ensemble learning comme la forêt aléatoire"


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


"validation croisée sur la regression linéaire"

# Définition du modèle de régression linéaire
model = LinearRegression()

# Validation croisée pour évaluer la performance du modèle
cv_scores = cross_val_score(model, x_train_const, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)

# Affichage des scores de validation croisée
print("Scores de RMSE pour chaque fold de la validation croisée : ", cv_rmse_scores)
print("RMSE moyen de la validation croisée : ", cv_rmse_scores.mean())


"comparaison de la validation croisée de la regression linéaire et la forêt aléatoire "

"OPTIMISATION DES HYPERPARAMETRES POUR AMELIORER LA ROBUSTESSE DES MODELES"


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


"VALIDATION CROISEE APRES OPTIMISATION DES HYPERPARAMETRES"


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