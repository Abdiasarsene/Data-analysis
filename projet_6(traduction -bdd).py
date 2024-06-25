# Importation des librairies
import pandas as pd

# Importation de la base de données
sav = pd.read_spss(r'c:\Users\ARMIDE Informatique\Downloads\Validiteit_SDQ_2023-2024.sav')
print(sav.count())
# vas = pd.read_spss(r'c:\Users\ARMIDE Informatique\Downloads\Betrouwbaarheid_SDQ_2023-2024.sav')

# Exportation de la base de données 
sav.to_excel('sav.xlsx', index= False)

# # Supression des données manquantes
# sup_sav = sav.dropna(axis=0)
# sup_vas = vas.dropna(axis = 0)

# # Affichage des données
#     # Première base de données* 
# print(sav.count())
# print(sav.count())

#     # Deuxième base de données
# print(vas.count())
# print(sup_vas.count())

# # Exportation de ma base donnée au format xlsx
# sup_sav.to_excel('spssnl.xlsx', index=False)
# sup_vas.to_excel('nldataset.xlsx', index=False)

# try :
#     # Importation de la base de données traduites
#     sav_fr = pd.read_excel(r'c:\Users\ARMIDE Informatique\Downloads\neerspss-français.xlsx')
#     vas_fr = pd.read_excel(r'c:\Users\ARMIDE Informatique\Downloads\nldataset-français.xlsx')

#     # Exportation de ma base donnée au format dta
#     sav_fr.to_stata('neerspss-français.dta')
#     vas_fr.to_stata('nldataset-français.dta')
# except Exception as e :
#     print('Erreur', e)
