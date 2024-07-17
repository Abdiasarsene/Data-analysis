# importation des librairies 
import pandas as pd

# importation de la base de données
an =pd.read_excel(r'c:\Users\ARMIDE Informatique\Documents\anl.xlsx')

an['MALE'].value_count()