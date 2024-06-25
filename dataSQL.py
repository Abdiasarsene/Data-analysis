# Importation des librairies
import sqlite3
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Chemin vers le fichier SQLite
chemin_db = "C:/Users/ARMIDE Informatique/Desktop/Formation pratique/data_immo.sqlite3"

# Connexion à la base de données SQLite
conn = sqlite3.connect(chemin_db)

# Création d'un curseur
cursor = conn.cursor()

try:
    # Exemple de requête SQL pour récupérer toutes les lignes de la table
    cursor.execute("SELECT valeur_fonciere_actuelle, code_voie FROM bien_immo")

    # Récupération des résultats
    rows = cursor.fetchall()
    drows = pd.DataFrame(rows, columns =("valeur_fonciere_actuelle","code_voie"))

    # Affichage des résultats "(pour l'exemple)
    for row in drows:
        print(' ')
        print(drows)

except sqlite3.Error as e:
    print("Erreur lors de l'exécution de la requête SQL :", e)

# Statistique descriptive
drows_stat = drows.describe()
print(' ')
print('RESULTATS DES ANALYSES STATISTIQUES EFFECTUEES')
print('')
print('Statistique descriptive')
print(drows_stat)
print(' ')

# test de Khi-deux
    # choix des variables 
col1 = 'valeur_fonciere_actuelle'
col2 = 'code_voie'

# Définir comme des df
data1 = drows[col1]
data2 = drows[col2]

# effectuer le test 
contingency = pd.crosstab(data1, data2)
chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency)

# Niveau de signification
alpha = 0.05

# Conclusion du test d'hypothèse
if p_value < alpha :
    print('Test de Khi-deux')
    print("Il y a une relation significative entre les deux variables (rejeter l'hypothèse nulle)")
else : 
    print("Il n'y a pas suffisamment de preuves pour rejeter l'hypothèse nulle")

# Régression linéaire
# Choix des varibales économiques
x = data1
y = data2

# Ajout de constant à la variable indépendante
x = sm.add_constant(x)

# Régression linéaire
model = sm.OLS(y, x).fit()
predictions = model.predict(x)

# Affichage des résultats
print(' ')
print('RESULTATS DES ANALYSES RÉGRESSIONNELLES EFFECTUEES')
print('')
print('Résultats de la régression linéaire')
print(model.summary())

# Affichage des résultats de la prédiction
print('')
print('Résultats de la prédiction')
print(predictions)
print(model.conf_int())

# Selection des variables
x = data1
y = data2

# Création du nuage de points (scatter plot)
# plt.bar(x, y)
plt.scatter(x, y)

# Ajout de titres et d'étiquettes
plt.title('Visualisation de x et y')
plt.xlabel('x')
plt.ylabel('y')

# Affichage du graphique
plt.show()

# Supposons que vous ayez un DataFrame df contenant vos données
# Par exemple, si vous souhaitez visualiser les corrélations entre toutes les colonnes de votre DataFrame :
corr_drows = drows.corr()

# Création de la heatmap
sns.heatmap(corr_drows, annot=True, cmap='coolwarm', fmt=".2f")

# Ajout de titres
plt.title('Matrice de corrélation')

# Affichage de la heatmap
plt.show()

# Supposons que vous ayez une colonne 'X' et une colonne 'Y' dans votre DataFrame df
X = data1
Y = data2

