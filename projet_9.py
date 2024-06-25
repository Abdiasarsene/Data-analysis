import pandas_datareader as pdr
import html5lib
import pandas as pd

# Utilisation de pandas_datareader pour récupérer des données financières
stock_data = pdr.get_data_yahoo('AAPL', start='2022-01-01', end='2022-12-31')

print("Données sur les stocks :")
print(stock_data.head())

# Utilisation de html5lib pour lire des données à partir d'une page web
url = 'https://fr.wikipedia.org/wiki/Liste_de_pays_par_PIB_(PPA)'
tables = pd.read_html(url)

print("\nTables sur la liste des pays par PIB (PPA) :")
for i, table in enumerate(tables):
    print(f"\nTable {i+1}:")
    print(table.head())
