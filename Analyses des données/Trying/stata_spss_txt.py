import pandas as pd

# Importation d'une base de données text
# txt = pd.read_csv(r"c:\Users\ARMIDE Informatique\Downloads\craxpro_db_01-03.txt")
# print(txt)

#Importation d'une base de données STATA
stata = pd.read_stata(r"c:\Users\ARMIDE Informatique\Desktop\Projet Académique\Projet personnel\Mémoire Abdias-Rodrigue\BDD_tourism.dta")

statas = stata.notna()
sta = pd.DataFrame(statas, columns =["Nuitees", "Taux_change"])
print(sta)
# sm = stata['Taux_croissanceco'].describe()
# print(sm)
# statas = pd.DataFrame(stata, columns =['Nbre_touriste','Taux_croissanceco','Depense_tourist'])
# #Statistique descriptive de la base de données
# statas_describ = statas.describe()

# statas_cov = statas.cov()
# statas_corr = statas.corr()

# # #Affichage des résultats
# print(statas_describ)
# print(statas_cov)
# print(statas_corr)

# #Importation d'une base de données SPSS
# spps = pd.read_spss(r"c:\Users\ARMIDE Informatique\Desktop\Formation pratique\Data Exo.sav")
# print(spps)