# Importation des librairies 
import pandas as pd
import numpy as np

# Définir les noms des provinces
provinces = ['British Columbia']

provinces = ['Alberta', 'Ontario', 'Québec', 'Nouvelle-Écosse', 'Manitoba']
# Définir les années
annees = range(1990, 2020)

# Générer des données aléatoires pour : 
    # le prix du carbone : 
prix_carbone = pd.DataFrame(np.random.uniform(5, 50, (len(annees), len(provinces))), index=annees, columns=provinces)

    # et l'investissement en énergie renouvelable

# Générer des données aléatoires pour le prix du carbone et l'investissement en énergie renouvelable
prix_carbone = pd.DataFrame(np.random.uniform(5, 50, (len(annees), len(provinces))), index=annees, columns=provinces)

investissement_energie_renouvelable = pd.DataFrame(np.random.uniform(100000, 1000000, (len(annees), len(provinces))), index=annees, columns=provinces)

# Afficher les cinq premières lignes des données générées
print("Prix du carbone:")
print(prix_carbone.head())
print("\nInvestissement en énergie renouvelable:")
print(investissement_energie_renouvelable.head())

# Exportation des données 
prix_carbone.to_excel ('bc.xlsx', index = True)
investissement_energie_renouvelable.to_excel('columbia.xlsx', index =True)
prix_carbone.to_excel ('prix.xlsx', index = True)
investissement_energie_renouvelable.to_excel('investissement.xlsx', index =True)
