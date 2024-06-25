#Importation de la base de donnée
import pandas as pd
from statsmodels.tsa.api import VAR
import sys

# Chargement de la donnée
try :
    bdd = pd.read_stata(r'Bases de données/BDD_tourisme.dta')
    # bdd = bdd.pop('Annees')
except Exception as e: 
    print("Erreur : ", e)

# # Méthode des moindres carrées
try:
    ##Méthode des moindres carrées
    model = VAR(bdd)
    result = model.fit()

        ##Affichage des résultats
    print(result.summary())
except Exception as e : 
    print('Erreur :',e)