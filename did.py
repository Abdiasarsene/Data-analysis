import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Données fictives
data = {
    'Year': [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009],
    'Group': ['Control']*5 + ['Treated']*5,
    'Outcome': [100, 102, 104, 106, 108, 110, 112, 114, 116, 118],
    'Post': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)
df.to_excel("diff.xlsx", index=False)

# Modèle de régression
model = ols('Outcome ~ Group * Post', data=df).fit()
print(model.summary())

# Effet de l'intervention
effect = model.params['Group[T.Treated]:Post']
print("Effet de l'intervention:", effect)
