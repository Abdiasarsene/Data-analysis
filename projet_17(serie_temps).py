import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
data_xlxs = pd.read_excel(r'c:\Users\ARMIDE Informatique\Desktop\Projet Académique\Projet personnel\Mémoire Abdias-Rodrigue\Base de données mémoire RETENUE.xlsx')
print(data_xlxs)

#Statistique descriptive
print(data_xlxs.describe())

#Tes de Dickey Fuller Augmenté
    ##Configurezn la colonne temporelle en tant qu'index
data_xlxs['Data'] = pd.to_datetime(data_xlxs['Annees'])
data_xlxs.set_index('Annees', inplace = True)

    #Selectionner la colonne de la serie temporelle à tester
ts = data_xlxs['Taux_croissanceco']

    #Effectuer le test de Dickey Fuller Augmenté
result = adfuller(ts)

    #Afficher les résultats
print('Statistique de test:', result[0])
print('p-value:', result[1])
print('28', result[3])
print('valeurs critiques:')
for key, value in result[4].items():
        print(f'(key):{value}')
        
#Effectuer le test de cointégration de Johansen
result = coint_johansen(data_xlxs, det_order=0, k_ar_diff=1)
order = result.ar_order

#Afficher les résultats
print("Statistique du test de cointégration de Johansen:")
print("Trace de la racine caractéristique (eigenvalues):", result.lr1)
print("Statistique du test de maximisation du rapport de vraisemblance (trace statistic):", result.lr2)
print("Seuils de signification (critical values):")
print(result.cvt)
print("Nombre de vecteurs de cointégration:", result.ind)

#Estimation du modèle VAR
model = VAR(data_xlxs)
results = model.fit(order)

#Affichage des résultats
print(results.summary())