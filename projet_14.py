# Importation des librairies importantes
import sqlite3 as sq
import pandas as pd

# Importation de la base de données 
eCommerce = r"c:\Users\ARMIDE Informatique\Desktop\Projet Informatique\BDD SQLite\commerceElectronique.sqlite"

# Connexion à la base de données 
connec = sq.connect(eCommerce)

# Création d'un curseur
cursor = connec.cursor()

# Rézcupération et affichage des données 
try:
    cursor.execute("SELECT nom, prenom, email, telephone FROM Clients")
    rows = cursor.fetchall()
    database_ecommerce = pd.DataFrame(rows, columns =(("nom","prenom","email","telephone")))
    print(database_ecommerce)
except Exception as e : 
    print(e)

finally : 
    connec.close()

# Exportation de la base de données sous un format xlsx
database_ecommerce.to_excel("database_ecommerce.xlsx", index=False)

# Connexion d'une nouvelle base de données
# Importation de la base de données 
eCommerce = r"c:\Users\ARMIDE Informatique\Desktop\Projet Informatique\BDD SQLite\commerceElectronique.sqlite"

# Connexion à la base de données 
connec = sq.connect(eCommerce)

# Création d'un curseur
cursor = connec.cursor()

try : 
    cursor.execute("SELECT prix, quantiteDeStock, description,evaluationMoyenne FROM Produits")
    ligne = cursor.fetchall()
    Produits_baseDeDonne = pd.DataFrame(ligne, columns=(("prix","quantiteDeStock","description","evaluationMoyenne")))
    print(Produits_baseDeDonne)
except Exception as e:
    print(e)

finally : 
    connec.close()

# Exportation de la table Commande en xlsx
# eCommerce = r"c:\Users\ARMIDE Informatique\Desktop\Projet Informatique\BDD SQLite\commerceElectronique.sqlite"

# Connexion à la base de données 
connec = sq.connect(eCommerce)

# Création d'un curseur
cursor = connec.cursor()

# Selection des colonnes
try:
    cursor.execute("SELECT statut,totalDeCommande, dateDeCommande FROM Commandes")
    index = cursor.fetchall()
    Commande = pd.DataFrame(index, columns=(('statut','totalDeCommande','dateDeCommande')))
    print(Commande)
except Exception as e:
    print(e)
finally: 
    connec.close()

# Table : Paiement
# Importation de la base de données
commerce = r'c:\Users\ARMIDE Informatique\Desktop\Projet Informatique\BDD SQLite\commerceElectronique.sqlite'

# Connexion à la base de données
conn = sq.connect(commerce)

# Création du cursor
cursor = conn.cursor()

# Récupération et affichage des données
try:
    cursor.execute('SELECT * FROM Paiements')
    iloc = cursor.fetchall()
    paiement =pd.DataFrame(iloc, columns=(('numeroDeCommande','methodeDePaiement','dateDePaiement','montantDePaiement')))
    print(paiement)
except Exception as e:
    print(e)
finally:
    conn.close()

# Table : Suivis de colis 
commer =r'c:\Users\ARMIDE Informatique\Desktop\Projet Informatique\BDD SQLite\commerceElectronique.sqlite'

# Connexion à la base de donées
con = sq.connect(commer)

# Création du cursor
cursor = con.cursor()

# Récupération et affichage des données
try:
    cursor.execute('SELECT * FROM SuivisDeColis')
    rows =cursor.fetchall()
    suivisdecolis = pd.DataFrame(rows, columns=(('numeroDeCommande','numeroDeColis','dateDeSuivi','dateDeReception')))
    print(suivisdecolis)
except Exception as e:
    con.close()
finally:
    con.close()

# Ouverture de plusieurs bases de données
Produits = pd.read_stata("Produits_ecommerce.dta")
Clients = pd.read_excel("database_ecommerce.xlsx")
Commandes= pd.read_csv("command.csv")
Paiements= pd.read_excel('paiement.xlsx')
SuivisDeColis= pd.read_excel('suivisdecolis.xlsx')

result = pd.concat([Clients,Produits, Commandes, Paiements, SuivisDeColis], axis=1 )

result.columns()

# Supression  des colones inutiles dans la base de données
commerceElectro = result.drop( columns=['index','Unnamed: 0','Unnamed: 0','numeroDeCommande','numeroDeCommande'])