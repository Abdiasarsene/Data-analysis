#Importation de la base de donnée
import pandas as pd
import statsmodels.api as sm
bdd = pd.read_excel(r'c:\Users\ARMIDE Informatique\Desktop\Formation pratique\BD zone CFA.xlsx')
print(bdd)

#Suppression de la datatime
sup1 = bdd.pop('dates')
print(sup1)

#Statistique descriptive
print(bdd.describe())

# Méthode des moindres carrées

    ## Selections des varaibles d'étude
x = bdd['IDE']
y = bdd['XAF']

    ##Ajoutez une constatnte à la var indep
x = sm.add_constant(x)

    ##Création du model
model = sm.OLS(y,x).fit()

    ##Affichage des résultats
print(model.summary())