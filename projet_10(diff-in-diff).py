# Importation des librairies
import pandas as pd 
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

# Importation de la base de données
DiD = pd.read_excel(r'c:\Users\ARMIDE Informatique\Downloads\diff-in-diff.xlsx')
# Affichage du contenu de la base de données
print(DiD)

# Calcul de la moyenne en agrégation
print('METHODE DIFF-IN-DIFF')
print('')
print('Moyenne des groupes avant et après')
print('')
mean_did = DiD.groupby('Groupe').mean()
print(mean_did)
print('')
mean_did = DiD.groupby('Groupe').mean()
print(mean_did)

# Vérifiez en calculant directement la moyenne pour chaque groupe.
    # 0 groupe de contrôle (Manitoba, Nova scotia, Ontario), 
    # 1 groupe de traitement (Alberta, Québec, British Comlubia)

mean_CO2eq_group0_0 = DiD.groupby('Groupe').mean().iloc[0, 0]
mean_CO2eq_group0_1 = DiD.groupby('Groupe').mean().iloc[0, 1]
mean_CO2eq_group1_0 = DiD.groupby('Groupe').mean().iloc[1, 0]
mean_CO2eq_group1_1 = DiD.groupby('Groupe').mean().iloc[1, 1]

# Affichage des résultats de la moyenne
print(f'Moyenne du groupe de contrôle avant: {mean_CO2eq_group0_0:.2f}')
print(f'Moyenne du groupe de contrôle après: {mean_CO2eq_group0_1:.2f}')
print(f'Moyenne du groupe de traitement avant: {mean_CO2eq_group1_0:.2f}')
print(f'Moyenne du groupe de traitement après: {mean_CO2eq_group1_1:.2f}')

#  Calcul de la méthode diff-in-diff
groupe0_diff = mean_CO2eq_group0_1 - mean_CO2eq_group0_0
groupe1_diff = mean_CO2eq_group1_1 - mean_CO2eq_group1_0
did = groupe1_diff - groupe0_diff

# Affichage du résultat
print('')
print(f'DID de la moyenne de GES est {did:.2f}')

# Création d'une nouvelle base de données
try :
    # Base de données de 2006

    DiD_2006 = DiD[['CO2eq_2006', 'Groupe']]
    DiD_2006['t'] = 0
    DiD_2006.columns = ['CO2eq', 'g', 't']

    # Base de données de 2015

    DiD_2015 = DiD[['CO2eq_2015', 'Groupe']]
    DiD_2015['t'] = 1
    DiD_2015.columns = ['CO2eq', 'g', 't']

    # Base de données de regression

    DiD_reg = pd.concat([DiD_2006, DiD_2015])

    # Creationb d'interaction

    DiD_reg['gt'] = DiD_reg['g'] * DiD_reg['t']
    print(DiD_reg)
except Exception as e:
    print('Erreur:',e)

# Créer un dictionnaire contenant les coefficients estimés
coef_dict = {
    'Variable': ['Intercept', 'g', 't', 'gt'],
    'Coefficient': [79.6710, 50.1641, -12.7271, 21.8183]
}

# Créer un DataFrame pandas à partir du dictionnaire
coef_df = pd.DataFrame(coef_dict)

# Afficher les coefficients
print(coef_df)

# Modélisation
# Selection des variables d'étude
X = DiD_reg[['g', 't', 'gt']]
y = DiD_reg.CO2eq

# Regression linéaire
# Ajout des constants
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

print('TEST DE DIAGNOSTIC')
print('')

# Test de diagnostique 
# Calculer les résidus
residuals = model.resid

# Tester l'hétéroscédasticité des résidus
lm, lm_p_value, fvalue, f_p_value = het_breuschpagan(residuals, X)

# Afficher les résultats
print("Hétéroscédasticité des résidus) :")
print("LM Statistic :", lm)
print("LM p-value :", lm_p_value)
print("F-Statistic :", fvalue)
print("F p-value :", f_p_value)

# Spécifiez votre modèle Diff-in-Diff
formula = 'CO2eq ~ g + t + gt'
model = ols(formula, data=DiD_reg).fit()

# Imprimez le résumé du modèle pour évaluer les résultats initiaux
print(model.summary())

# Test de sensibilité : Modifier la spécification du modèle
# Par exemple, en ajoutant des variables de contrôle ou en modifiant les périodes de temps
formula_sensitivity = 'CO2eq ~ g + t + gt + X1 + X2'

# Ajoutez des variables de contrôle X1 et X2
model_sensitivity = ols(formula_sensitivity, data=DiD_reg).fit()

# Affichage du résulats
print(model_sensitivity.summary())

# Test de robustesse : Utilisez différents groupes de contrôle
# Par exemple, en changeant les groupes de contrôle ou en ajustant les groupes de traitement
formula_robustness = 'CO2eq ~ g_alt + t_alt + gt_alt'
model_robustness = ols(formula_robustness, data=DiD_reg).fit()

# Affichage du résultat
print(model_robustness.summary())
print(model_robustness.summary())