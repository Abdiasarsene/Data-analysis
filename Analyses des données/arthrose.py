# Importation des bibliothèques
import pandas as pd
import missingno as msno

# importation de la base de données
donne_arthrose = pd.read_excel(r'Bases de données/DonneSante.xlsx')
donne_arthrose.head() # afficher les cinq premières lignes
donne_arthrose.tail() # afficher les cinq dernières lignes
donne_arthrose.isnull().sum() #afficher les valeurs manquantes de la base de données
donne_arthrose.duplicated.sum() #afficher les doublons
donne_arthrose.bar() #afficher graphiquement les données manquantes de la base de données

#Analyse exploratoire des données 
