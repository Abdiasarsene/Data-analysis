# Importation des bibliothèqyues
import pandas as pd
import statsmodels.api as sm

# Importation de la base de données
data1 = pd.read_excel(r'Bases de données/Base de données mémoire RETENUE.xlsx')

try :
    # Selections des variables économiques
        # Variable dépendante
    col1 = 'Taux_croissanceco'
    x = data1[col1]

        # Variables indépendantes
    col2 = 'Nbre_touriste'
    col3 ='Depense_tourist'
    col4 = 'Taux_change'
    y1 = data1[col2]
    y2 = data1[col3]
    y3 = data1[col4]
    y = y1, y2, y3
    x= pd.to_numeric(x, IndexError())
    y= pd.to_numeric(y, IndexError())

    # Ajout des constants à la variable dépendante
    x = sm.add_constant(x)

    # Regression linéaire multiple
    model = sm.OLS(x, y).fit()

    # Affichage des résultats de la regression
    print(model.summary)
except Exception as e : 
    print('Erreur : ', e)