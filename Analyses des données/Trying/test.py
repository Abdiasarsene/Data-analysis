import yfinance as yf

# Symbole pour le CFA franc de l'Afrique de l'Ouest (XOF)
xof = yf.Ticker('X')

# Symbole pour le CFA franc de l'Afrique centrale (XAF)
xaf = yf.Ticker('X')

# Récupération des données sur le CFA franc de l'Afrique de l'Ouest
data_xof = xof.history(period="1y")  # Vous pouvez ajuster la période comme vous le souhaitez

# Récupération des données sur le  CFA franc de l'Afrique centrale
data_xaf = xaf.history(period="1y")  # Vous pouvez ajuster la période comme vous le souhaitez

# Affichage des données
print("Données XOF (Afrique de l'Ouest) :")
print(data_xof)
print("\nDonnées XAF (Afrique centrale) :")
print(data_xaf)
