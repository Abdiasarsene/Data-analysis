#Importation de la BDD
import sqlite3
#Data haNDLING
    #Data deleting
sup1 = data_xlxs.pop('region')
sup2 = data_xlxs.pop('timestamp')
sup3 = data_xlxs.pop('country')
sup4 = data_xlxs.pop('gender')
print(data_xlxs)

    #Suppression des valeurs manquantes
data_xlxs = data_xlxs.dropna()
data_sql = sqlite3.read_sql(r'\\c\Users\ARMIDE Informatique\Desktop\Formation pratique\panamapapers.sqlite3')