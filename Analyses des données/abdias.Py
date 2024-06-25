import pandas as pd
import statsmodels.api as sm

#Importation de la base de donnée
abdias =pd.read_excel(r"c:\Users\ARMIDE Informatique\Desktop\Formation pratique\bddias.xlsx")
print(abdias)

#Suppression de la variable 'date'
delp = abdias.pop('date')
print(delp)

#Renommer une variable
renab = abdias.rename(columns= {'indice_prix_logement':'idc_prix'})
print(renab)
print(abdias)

#Statistique descriptive
print(abdias.describe())

#Regression linéaire
    #Selectioner les variabls pertinentes
x = abdias['indice_prix_logement']
y = abdias['valeur_fonciere']

#Ajouter la constant à la variable dép
X = sm.add_constant(x)

#Créer le modèle
model = sm.OLS(y,x).fit()

#Afficher les résultats
print(model.summary())