#Importation d'une BDD pour le tableau 
import pandas as pd
from sklearn.linear_model import LinearRegression
tourism = pd.read_excel(r'C:\Users\ARMIDE Informatique\Desktop\Projet Académique\Projet personnel\Mémoire Abdias-Rodrigue\Base de données mémoire RETENUE.xlsx')

#manipulation des données
    ## Insertion d'une nouvelle variable
Nut = tourism['Nnu'] = tourism['Depense_tourist']/2
print(Nut)
    ##Supprimer des variables
delnuitee = tourism.pop('Nuitees')
delnu2 = tourism.pop('Recette globale')
delnu3 = tourism.pop('Nnu')
print (tourism)

#Statistique descriptive 
print("statistique descriptive de toutes lesvariables :" )
print(tourism.describe())

# Sélectionnez les colonnes pertinentes
X = tourism['Taux_croissanceco'].values.reshape(-1, 1)
y = tourism['Depense_tourist'].values

# Créez le modèle de régression linéaire
model = LinearRegression()

# Ajustez le modèle aux données
model.fit(X, y)

# Prédiction avec le modèle
y_pred = model.predict(X)

# Affichage des résultats
print("Coefficient de pente (pente de la régression) :", model.coef_[0])
print("Terme constant (ordonnée à l'origine) :", model.intercept_)

#Tester la significativité
import statsmodels.api as sm

# Sélectionnez les colonnes pertinentes
x = ['Nbre_touriste', 'Depense_tourist']
y = ['Taux_croissanceco']

# Ajoutez une constante à la variable indépendante
X = sm.add_constant(X)

# Créez le modèle de régression linéaire
model = sm.OLS(y, X).fit()

# Affiche un résumé des résultats de la régression
print(model.summary())
