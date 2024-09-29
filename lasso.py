# Importation des librairies
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso

#Importation de la base de données
bd = pd.read_excel(r"d:\Projet Informatique\Bases de données\donnees_sante.xlsx")
dataset = bd.select_dtypes('int')

# Détection des valeurs abérrantes
Q1 = dataset.quantile(0.25)
Q3 = dataset.quantile(0.75)
IQR = Q3 - Q1
outliers =((dataset<(Q1- 1.5*IQR))|(dataset>(Q3 + 1.5*IQR)))
valeurs_aberrantes = dataset[outliers.any(axis=1)]

if not dataset.empty :
    print("Jeu de données sans danger")
else:
    print("La base de données contient des valeurs abérrantes", valeurs_aberrantes)

# Vérification des doublons dans le jeu de données
if dataset.duplicated().sum() :
    print("Le jeu de données est sans doublon")
else :
    print("Une base de données avec des doublons")

# Normalisation des données
scaler = MinMaxScaler()
variables = ['age', 'taux_de_cholesterol', 'taux_de_glycemie','pression_arterielle']
datanorma = dataset[variables]
datanorm=scaler.fit_transform(dataset[variables])
datanorma

# Préparation des variables d'études
x= datanorma.drop(columns=['age'])#Prédicteurs
y= datanorma['age'] # Variable cible

# Division des données en entraînement et en test
x_train, x_test,y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=808)

# Ajout des constants
x_train_const = sm.add_constant(x_train)
x_test_const = sm.add_constant(x_test)

# Entraînement du modèle
model = sm.OLS( y_train, x_train_const).fit()

#Prédiuction
y_pred =model.predict(x_test_const) 

# Métric de performance
print('R² :', r2_score(y_pred, y_test))
print('MAE :', mean_absolute_error(y_pred, y_test))
print('RMSE :', mean_squared_error(y_pred, y_test))

# Utilisation de la forêt aléatoire
#Initialisation et l'entraînement du modèle
rf_model = RandomForestRegressor(n_estimators =100, random_state=42)
rf_model.fit(x_train_const, y_train)

# Prédiction
y_pred_rf = rf_model.predict(x_test_const)

# Métrics de performance
print('R² :', r2_score(y_pred_rf, y_test))
print('MAE :', mean_absolute_error(y_pred_rf, y_test))
print('RMSE :', mean_squared_error(y_pred_rf, y_test))

# Utilisation Lasso et Ridge
ridge_mod = Ridge(alpha=1.0)
lasso_mod = Lasso(alpha=0.1)
ridge_mod.fit(x_train_const, y_train)
lasso_mod.fit(x_train_const, y_train)

# Prédictions
y_pred_rd = ridge_mod(x_test_const)
y_pred_las = lasso_mod(x_test_const)

# Métrics de performance
# LASSO
print("Lasso MAE ;", mean_absolute_error(y_pred_las, y_test))
print("Lasso R² ; ",r2_score(y_pred_las))
print("Lasso RMSE :", mean_squared_error(y_pred_las, y_test))

# RIDGE
print("Lasso MAE ;", mean_absolute_error(y_pred_rd, y_test))
print("Lasso R² ; ",r2_score(y_pred_rd))
print("Lasso RMSE :", mean_squared_error(y_pred_rd, y_test))

# Entraînement des modèles
ridge_mod = Ridge(alpha=1.0)
lasso_mod = Lasso(alpha=0.1)
ridge_mod.fit(x_train_const, y_train)
lasso_mod.fit(x_train_const, y_train)

# Prédictions
y_pred_rd = ridge_mod.predict(x_test_const)
y_pred_las = lasso_mod.predict(x_test_const)

# Métriques de performance
# LASSO
print("Lasso MAE :", mean_absolute_error(y_test, y_pred_las))
print("Lasso R² :", r2_score(y_test, y_pred_las))
print("Lasso RMSE :", mean_squared_error(y_test, y_pred_las, squared=False))

# RIDGE
print("Ridge MAE :", mean_absolute_error(y_test, y_pred_rd))
print("Ridge R² :", r2_score(y_test, y_pred_rd))
print("Ridge RMSE :", mean_squared_error(y_test, y_pred_rd, squared=False))