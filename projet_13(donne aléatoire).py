# Importation des données
import pandas as pd
import numpy as np

# Définir le nombre de lignes à générer
n = 100

# Générer des données aléatoires
data = {
    "Arthrose": np.random.choice(['oui', 'non'], n),
    "age": np.random.randint(20, 91, n),
    "taux_de_cholesterol": np.random.randint(150, 301, n),
    "taux_de_glycemie": np.random.randint(70, 151, n),
    "Medicament": np.random.choice(['Doliprane', 'Simvastatine', 'Rosuvastatine'], n),
    "pression_arterielle": np.random.randint(90, 181, n)
}

# Créer un DataFrame
df = pd.DataFrame(data, columns=(("Arthrose","age","taux_de_cholesterol","taux_de_glycemie","Medicament","pression_arterielle")))

# Creation d'une nouvelle variable par choix aléatoire
m = 100
donne = {"Sexe": np.random.choice(['Masculin', 'Feminin'],m)}
dat = pd.DataFrame(donne,columns=['Sexe'])
dat['Sexe'], codes = pd.factorize(dat['Sexe'])

# Fusion des deux bases de données
base1 = pd.read_excel('donnees_sante.xlsx')
base2= pd.read_excel('dat.xlsx')
base3 = pd.read_excel('sexe.xlsx')
DonneSante = pd.concat([base1, base2], axis=1)

# Supposons que vous ayez déjà votre objet DataFrame DonneSante

# Colonnes à coder
colonnes_a_coder = ['Medicament', 'Arthrose']

# Dictionnaire pour stocker les correspondances entre les catégories et les codes
codes = {}

# Boucle sur chaque colonne à coder
for colonne in colonnes_a_coder:
    # Appliquer pd.factorize() pour coder la colonne et récupérer les codes ainsi que les correspondances
    DonneSante[colonne + '_code'], codes[colonne] = pd.factorize(DonneSante[colonne])

# Renommer une colonne
DonneSante.rename(columns={'Sexe': 'Sexe'}, inplace=True)
DonneSante.drop(columns=['Unnamed: 0'])

datahealth = pd.concat([DonneSante, base3], axis=1)