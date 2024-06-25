import pandas as pd
from statsmodels.tsa.stattools import kpss

# Importez vos données dans un DataFrame Pandas (par exemple, à partir d'un fichier excel)
donnees = pd.read_excel("C:\Users\ARMIDE Informatique\Desktop\Projet Académique\BD zone CFA.xlsx")

# Spécifiez la colonne pour laquelle vous souhaitez effectuer le test de stationnarité
colonne_d_interet = "XAF"

# Effectuez le test de stationnarité de KPSS
resultat_kpss = kpss(donnees[colonne_d_interet])

# Extrayez les statistiques du test et la valeur critique

valeur_critique_5 = resultat_kpss[3]['5%']

# Affichez les résultats du test
print("Statistique du test de KPSS :", statistique_kpss)
print("Valeur critique à 5% :", valeur_critique_5)

# Comparez la statistique du test à la valeur critique pour prendre une décision

    print("Les données sont stationnaires au niveau de signification de 5%.")
elif statistique_kpss < valeur_critique_10:
    print("Les données ne sont pas stationnaires.")
