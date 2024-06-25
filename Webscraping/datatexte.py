import pandas as pd

datatexte = pd.read_csv(r"")
print(datatexte)


# Lire le fichier texte ligne par ligne
with open(r"c:\Users\ARMIDE Informatique\Desktop\datatexte.txt",  encoding="utf-8") as file:
    lines = file.readlines()

# Créer une DataFrame avec une colonne "Titre" contenant les lignes lues
df = pd.DataFrame({"Titre": lines})

# Afficher la DataFrame
print(df)
