# Importation des librairies d'études
import pandas as pd
from scipy import stats

# Importation de la base de donnéess sur Python
    # Sur Excel : 
bdata1 = pd.read_excel(r'Bases de données/bddias.xlsx')

# J'ai affiché les variables et le nombre d'observation de la  base de données avec cette commande.
print(bdata1.count())

try :
    # Je l'ai fait afin de voir les variables à utiliser pour réaliser mon test de Khi-deux : 
        # Dès lors selectionnons nos variables pour réaliser le test
    variable1 = bdata1['indice_prix_logement']
    variable2 = bdata1['valeur_fonciere']

        # Je réalise mon test de Khi-deux,mais avant tout, il faut que je réalise le tableau de contingence grâce à la commande : pd.crosstab
    contingency = pd.crosstab(variable1, variable2)

        # Pour maintenant réaliser le test de khi-deux, importons la librairie : from scipy import statS. ce qui a été importé au début de l'analyse.
        # Je réalise le test de khi-deux avec la commande : stats.chi2_contingency
    chi2, p, dof, expected = stats.chi2_contingency(contingency)

    # Passons immédiatemment à la conclusion de notre étude : 
        # # Mais, definisson le niveau de significativité pour la conclusion,
        # Niveau de significativité
    alpha = 0.05

        # Conclusion du test d'hypothèse
    if p < alpha :
        print('Test de Khi-deux')
        print("")
        print("Il y a une relation significative entre les deux variables (rejeter l'hypothèse nulle)")
    else : 
        print("")
        print("Il n'y a pas suffisamment de preuves pour rejeter l'hypothèse nulle")
except Exception as e:
    print('Erreur : ',e )