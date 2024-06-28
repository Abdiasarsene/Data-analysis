import pandas as pd
from scipy import stats
import statsmodels.api as sm

# Ajout de toutes les bases de données
data_1 = pd.read_excel(r'Bases de données/BD financement traditionnel.xlsx')
data_2 = pd.read_excel(r'Bases de données/Base_crowdfunding.xlsx')

# Concaténation
datas_concated = pd.concat([data_1, data_2], ignore_index = True)

# Constitution d'une nouvelle base de données
new_dataset = pd.DataFrame(datas_concated, columns =('Chiffre_Affaire','Montant_crowdfunding', 'Montant_FTraditionnel','Revenus_Nets'))

# Visualisation des données manquantes
isnadata = new_dataset.isna().any(axis = 1)

print(isnadata)

# Suppression des valeurs manquantes
drop_new_dataset = new_dataset.dropna(axis=0)

# Statistique descriptives et cov
stat_descri = drop_new_dataset.describe()
corr_drop_set = drop_new_dataset.corr()

# Visualisation des données
print(stat_descri)
print(corr_drop_set)

# Choix des variables de comparaison7
col1 = 'Montant_crowdfunding'
col2 =  'Montant_FTraditionnel'

data1 = datas_concated[col1]
data2 = datas_concated[col2]

# Effectuer un test t-student pour déterminer si les moyennes des deux ensembles sont significativement différentes
t_statistic, p_value = stats.ttest_ind(data1, data2)

# Niveau de signification
alpha = 0.05

# Vérifier si la valeur p est inférieure au niveau de signification alpha
if p_value < alpha:
    print("Les moyennes des deux ensembles sont significativement différentes (rejeter l'hypothèse nulle)")
else:
    print("Il n'y a pas suffisamment de preuves pour rejeter l'hypothèse nulle")

# le test de Khi-deux
contingency = pd.crosstab(data1, data2)

# Effectuez le test de khi-deux
chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency)

# Niveau de signification 
alpha = 0.05

# Conclusion
if p_value < alpha : 
    print ("Il y a une relation significative entre les deux variables (rejeter l'hypothèse nulle)")
else : 
    print("Il n'y a pas suffisamment de preuves pour rejeter l'hypothèse nulle")

# Regression linéaire
    # Selection des variables 
x = data1
y = data2

# Ajout des constants
x = sm.add_constant(x)

# Modèle de la regression
model = sm.OLS(y, x).fit()

# Affichafe des résultats
print(model.summary())
print(model.predict())
print(model.conf_int())
