# Importation des bibliothèques
import pandas as pd
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# importation de la base de données
donne_arthrose = pd.read_excel(r'c:\Users\ARMIDE Informatique\Desktop\Formation pratique\DonneSante.xlsx')

donne_arthrose.head() # afficher les cinq premières lignes
donne_arthrose.tail() # afficher les cinq dernières lignes
missindata = donne_arthrose.isnull().sum()
missindata#afficher les valeurs manquantes de la base de données
datadupli = donne_arthrose.duplicated().sum() 
datadupli#afficher les doublons

numericData = donne_arthrose.select_dtypes(exclude=['object']) #écarter les variables catégorielles dans la base de données
execptData =numericData.drop(columns=['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0']) #supprimer certaines variables inutiles dans la base de données

msno.bar(execptData, color='green') #afficher graphiquement les données manquantes de la base de données

#Analyse exploratoire des données 
statistic= execptData.describe()

# visualisation des données
dataexpect = donne_arthrose.drop(columns=['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0'])

#Histogramme et densité des variables
fig, ax =plt.subplots(1,2,figsize=(14,6))
sns.histplot(dataexpect['Arthrose_code'], kde=True, color='blue', ax=ax[0])
ax[0].set(
    title='Histogramme et densité des variables'
)

#Graphiques en barres pour les variables catégorielles
sns.countplot(x='Arthrose_code', data=dataexpect, palette='muted',color='purple')
ax[1].set(
    title='Graphiques en barres pour les variables catégorielles'
)

# Variables à tracer
variabl = ['pression_arterielle', 'Sexe.1']

# Création de la figure et des axes
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Tracé des box plots sur chaque sous-graphe
for ax, var in zip(axes, variabl):
    sns.boxplot(x='Arthrose_code', y=var, data=dataexpect, palette='pastel', ax=ax)
    ax.set_title(f'Box Plot of {var} by Arthrose_code')

# Ajustement de l'espacement entre les sous-graphes
plt.tight_layout()
plt.show()


# Variables à tracer
variables = ['age', 'taux_de_cholesterol', 'taux_de_glycemie', 'Medicament_code']

# Création de la figure et des axes
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Tracé des box plots sur chaque sous-graphe
for ax, var in zip(axes.flatten(), variables):
    sns.boxplot(x='Arthrose_code', y=var, data=dataexpect, palette='pastel', ax=ax)
    ax.set_title(f'Box Plot of {var} by Arthrose_code')

# Ajustement de l'espacement entre les sous-graphes
plt.tight_layout()
plt.show()

# préparation de la base de données
y= execptData['Arthrose_code']#variable cible
x= execptData[['age', 'taux_de_cholesterol', 'taux_de_glycemie', 'pression_arterielle',
'Sexe', 'Medicament_code']]#variables prédicteurs

# division des données en entraînement et en test
x_train, x_test, y_train, y_test=train_test_split(y,x,test_size=0.2, random_state=42)

# Exemple de chargement des données
y = execptData['Arthrose_code']  # Variable cible
X = execptData[['age', 'taux_de_cholesterol', 'taux_de_glycemie', 'pression_arterielle', 'Sexe', 'Medicament_code']]  # Variables prédictives

# entraînement du modèle
X = sm.add_constant(X)
model = sm.Logit(y_train, x_train)
result = model.fit()
print(result.summary())

# prediction
y_pred = model.predict(x_test)

# probabilité de la préduction
y_pred_proba = model.predict(x_test, transform=False)

# Histogramme de la probabilité de prédiction
plt.hist(y_pred_proba, bins=10, edgecolor='k')
plt.xlabel('Probabilité de prédiction')
plt.ylabel('Fréquence')
plt.title('Histogramme des probabilités de prédiction')
plt.show()

# évaluation de la performance prédictive
print('Accuracy:', accuracy_score(y_test, y_pred))

# Rapport de classification
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Matrice de confusion
print('Confusion Matrix:')
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Affichage de la matrice de confusion sous forme graphique
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Prédiction')
plt.ylabel('Réel')
plt.title('Matrice de confusion')
plt.show()

# Courbe ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Courbe ROC (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs (FPR)')
plt.ylabel('Taux de vrais positifs (TPR)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
