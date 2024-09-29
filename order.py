import pandas as pd

# Charger votre base de données existante
df = pd.read_excel(r'D:\Projet Académique\BDD et Results\bdd_canada.xlsx')  # Remplacez par le nom de votre fichier

# Définir les provinces traitées et les provinces de contrôle
treated_provinces = ['Alberta', 'British Columbia', 'Québec']  # Provinces avec tarification du carbone
control_provinces = ['Manitoba', 'Ontario', 'Nova Scotie']     # Provinces sans tarification

# Créer la variable 'Post' : 1 pour les années 2008 et après, 0 avant 2008
df['Post'] = df['Year'].apply(lambda x: 1 if x >= 2008 else 0)

# Créer la variable 'Treated' : 1 pour les provinces traitées, 0 pour les autres
df['Treated'] = df['Region'].apply(lambda x: 1 if x in treated_provinces else 0)

# Enregistrer la base de données avec les nouvelles colonnes dans un fichier Excel
df.to_excel('base_donnees_avec_post_treated.xlsx', index=False)

# Afficher les premières lignes pour vérifier
print(df.head())
