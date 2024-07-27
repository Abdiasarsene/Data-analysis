# Importation des librairies
import pandas as pd
import numpy as np

# Importation des bases de données
an = pd.read_excel(r'c:\Users\ARMIDE Informatique\Downloads\Combined ESG.xlsx')
mn = pd.read_excel(r'c:\Users\ARMIDE Informatique\Downloads\ESG Scores.xlsx')

# Nombre d'observations
n_obs = 200

# Fonction pour vérifier l'existence d'une colonne et générer des données si nécessaire
def get_column_or_generate(df, column_name, generator):
    if column_name in df.columns:
        return df[[column_name]]
    else:
        return pd.DataFrame({column_name: generator(size=n_obs)})

# Extraction ou génération des variables d'intérêt
fonctionnement_reflexif = get_column_or_generate(an, 'Maternal Reflective Functioning', lambda size: np.random.uniform(0, 10, size))
stress_prenatal = get_column_or_generate(an, 'Prenatal Stress', lambda size: np.random.uniform(0, 10, size))
support_social = get_column_or_generate(mn, 'Social Support', lambda size: np.random.uniform(0, 10, size))
relation_mere_enfant = get_column_or_generate(mn, 'Mother-Child Relationship Quality', lambda size: np.random.uniform(0, 10, size))
child_behavior_problems = get_column_or_generate(mn, 'Child Behavior Problems', lambda size: np.random.randint(0, 100, size))

# Combinaison des variables de la relation mère-enfant et des problèmes de comportement de l'enfant
relation_mere_enfant_problemes = pd.concat([relation_mere_enfant, child_behavior_problems], axis=1)

# Combinaison des données dans un seul DataFrame
df_final = pd.concat([fonctionnement_reflexif, stress_prenatal, support_social, relation_mere_enfant_problemes], axis=1)

# Affichage du DataFrame final
print(df_final.head())

# Exportation du DataFrame final en Excel si nécessaire
df_final.to_excel(r'c:\Users\ARMIDE Informatique\Downloads\final_data.xlsx', index=False)
