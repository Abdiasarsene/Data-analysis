# IMPORTATION DES LIBRAIIRES
import pandas as pd
import xlrd
import matplotlib.pyplot as ply
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# IMPORTATION DE LA BASE DE DONNEES
ro_data = pd.read_excel(r'c:\Users\HP ELITEBOOK 840 G6\Downloads\BASE DE DONNEES RECONSTITUEES.xlsx')

ro_data.columns #Affichage des variables de la base de données
ro_data.isna().sum() #Affichage des valeurs manquantes
ro_data.duplicated().sum() #Affichage des doublons

# Détection  des valeurs abérrantes
Q1 = ro_data.quantile(0.25)
Q3 = ro_data.quantile(0.75)
IQR = Q3 - Q1
aberrantes = ((ro_data<(Q1-1.5*IQR))|(ro_data>(Q3+1.5*IQR)))
outlier = ro_data[aberrantes.any(axis=1)]
outlier

# Correction des valeurs aberrantes en les remplaçant par la médiane
ro_data_corrige = ro_data.copy()
for col in ro_data.columns:
    ro_data_corrige.loc[aberrantes[col], col] = ro_data[col].median()

ro_data_corrige

# Impoortation de la base de données  qui contient les variantes de la variable "GPR"
gpr = pd.read_csv(r'c:\Users\HP ELITEBOOK 840 G6\Downloads\data_gpr_export.xls')
gpr.columns

# Extraction de variantes utiles pour la création de la variable "GPR"
variables = ['month', 'GPR', 'GPRT', 'GPRA', 'GPRH', 'GPRHT', 'GPRHA', 'SHARE_GPR', 'GPRHC_USA', 'GPRHC_UKR']

# Création d'un DataFrame avec les  variantes
data_gpr =pd.DataFrame(data =gpr, columns=variables)

# Remplissage des valeurs manquantes par la médiane du DataFrame
fillna_gpr = data_gpr.fillna(data_gpr.median)

# Création de la variable "gPR"
fillna_gpr['gPR'] = fillna_gpr[variables].mean(axis=1)

# Création d'un nouveau DataFrame pour stocker la variable seule et après le concatener avec la base de données "ro_data_corrige"
grp =pd.DataFrame(data= fillna_gpr, columns=['gPR'])

# Exportation des bases de données 
grp.to_excel('grp.xlsx')
ro_data_corrige.to_excel("ro_data.xlsx")
fillna_gpr.to_excel('gpr.xlsx')

# Importation de la base de données d'étude 
study_data = pd.read_excel(r'ro_data.xlsx')

# Tronquage de la base de données sur 2000 obs
nombre_dobservation = 2000
datacanada =study_data.head(nombre_dobservation)

# Supression des variables iunutiles
datacanada = datacanada.drop(columns=['Unnamed: 0'])

# Remplacer les valeurs string par zéro
datacanada['GPR'] = datacanada['GPR'].apply(lambda x: 0 if isinstance(x, str) else x)

# Normalisation des données
vars = ['log Lender_ShareN', 'LnTranche_Amount', 'Working_capital',
    'SecuredNew', 'Long_term', 'Short_term', 'log Number_of_Lenders',
    'ROA_BORROWER', 'Total_loans_to_total_assets_LEND',
    'Total_deposits_to_total_assets_L', 'ROA_LENDER', 'RefinancingNew',
    'Size_BORROWER', 'Number_of_Lead_Arrangers', 'GPR']
datacanada[vars] = MinMaxScaler().fit_transform(datacanada[vars])