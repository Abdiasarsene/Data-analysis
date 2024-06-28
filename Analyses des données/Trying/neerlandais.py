import pandas as pd
# neer = pd.read_excel(r'c:\Users\ARMIDE Informatique\Downloads\Statistiek Winford databestand (1).xlsx' , r'c:\Users\ARMIDE Informatique\Downloads\Statistiek Winford databestand (2).xlsx ')

# dfs = [pd.read_excel(file) for file in neer]
# # Concaténer les DataFrames
# concatenated_df = pd.concat(dfs, ignore_index=True)

# # Visualiser les données
# print(concatenated_df)
# print(neer.count)
# print(neer.dtypes)
# print(neer.shape)

4
import pandas as pd

# Lire les feuilles Excel
neer_sheet1 = pd.read_excel(r'c:\Users\ARMIDE Informatique\Downloads\Statistiek Winford databestand (1).xlsx')
neer_sheet2 = pd.read_excel(r'c:\Users\ARMIDE Informatique\Downloads\Statistiek Winford databestand (2).xlsx')

# Concaténer les feuilles
neer_concatenated = pd.concat([neer_sheet1, neer_sheet2], ignore_index=True)

# Visualiser le DataFrame concaténé
print(neer_concatenated.dtypes)
print(neer_concatenated.count)
print(neer_concatenated.shape)
