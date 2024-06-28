#Ipmportation de la BDD
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
tourisme = pd.read_excel(r'C:\Users\ARMIDE Informatique\Desktop\Projet Académique\Projet personnel\Mémoire Abdias-Rodrigue\Base de données mémoire RETENUE.xlsx')

#Statistique descriptive
print("statistique de toutes les variables")
print(tourisme.describe())

#Test de Dickey Fuller Augmenté
    # Configurez la colonne temporelle en tant qu'index
tourisme['Date'] = pd.to_datetime(tourisme['Annees'])  
    # Assurez-vous que la colonne de dates est au format datetime
tourisme.set_index('Annees', inplace=True)

# Sélectionnez la colonne de la série temporelle à tester
ts = tourisme['Taux_croissanceco']

# Effectuez le test de Dickey-Fuller augmenté
result = adfuller(ts)

# Affichez les résultats
print('Statistique de test :', result[0])
print('p-value :', result[1])
print('Lags utilisés :', result[2])
print('28 :', result[3])
print('Valeurs critiques :')
for key, value in result[4].items():
    print(f'   {key}: {value}')
    
#rEGRESSION LINEAIRE
    #Slectionner les variables pertinentes
x = ['Nbre_touriste', 'Depense_tourist']
y = ['Taux_croissanceco']

    #Ajouter une constante aux variables indépendates
x = sm.add_constant(x)

    #Création du modèle économétrique
model = sm.OLS(y,x).fit()

    #Affichage des résultats
print(model.summary())