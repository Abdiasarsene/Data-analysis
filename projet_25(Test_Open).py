# Importation des livrairies
import pandas as pd

# Importation de la base de données
open_url='https://raw.githubusercontent.com/OpenClassrooms-Student-Center/8063076-Initiez-vous-au-Machine-Learning/master/data/ADEME_dpe-v2-tertiaire-2.csv'

paris_url= 'https://github.com/OpenClassrooms-Student-Center/8063076-Initiez-vous-au-Machine-Learning/blob/master/notebooks/P4C2_robustesse_modele.ipynb'

openclassroom=pd.read_csv(open_url)
paris=pd.read_csv(paris_url)
