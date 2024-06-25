# Importation des librairies d'études
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns 

# Importation de la base de donnéess sur Python
    # Sur Excel : 
bdata1 = pd.read_excel(r'Bases de données/bddias.xlsx')
print(bdata1)

# Statisitique 
stat = bdata1.describe()
print(stat)

# covariance 
cova = bdata1.cov()
print(cova)

# et corrélation
corre = bdata1.corr()
print(corre)

    # Sur csv : 
bdata2 = pd.read_csv(r'Bases de données/clientele.csv')
print(bdata2)

    # Sur stata : 
bdata3 = pd.read_excel(r'Bases de données/BDD dpae-annuelles-france.dta')
print(bdata3)

    # Sur spss
bdata4 = pd.read_csv(r'Bases de données/clientele.csv')
print(bdata4)

# Définition des variables
col1 = 'indice_prix_logement'
col2 = 'valeur_fonciere'
# indice_logement = bdata[col1]
# valeur_fonciee = bdata[col2]

sns.scatterplot(x = col1, y = col2)
plt.show()
# # Affichage des graphiques
# plt.scatter(indice_logement, valeur_fonciee)

# # Visualisation des données
# plt.title('Nuage de point')
# plt.xlabel('indice_logement')
# plt.ylabel('valeur_fonciere')

# # Affichage des resultats
# plt.show()