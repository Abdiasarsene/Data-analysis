import pandas as pd
canada=pd.read_excel(r'd:\Projet Informatique\Bases de données\data_canada.xlsx')
canada = canada.select_dtypes('int')
canada.to_excel('canada_data.xlsx', index=False)

canada.columns

# List of variable names based on the provided details
variable_names = [
    'Proportion_prêts_retenue', 'Maturité_prêt', 'Risque_géopolitique', 'Log_montant_transaction',
    'Garantie', 'Fonds_de_roulement', 'Acquisition', 'Refinancement', 'Log_actifs_t',
    'ROA_t-1_emprunteur', 'Valeur_marché_t-1', 'Nombre_chefs_de_file', 'Log_nombre_prêteurs',
    'Log_taille_t-1', 'Ratio_capital_t-1', 'ROA_t-1_prêteur', 'Total_dépôts_actifs_t-1',
    'Total_prêts_actifs_t-1', 'Années'
]

# Rename the first columns based on the variables (adjusting for the actual number of columns)
# and keeping only those columns with known variable names.
num_variables = len(variable_names)
data_renamed = data.iloc[:, :num_variables]
data_renamed.columns = variable_names

# Removing unnamed columns (those beyond the specified variables)
cleaned_data = data_renamed

# Save the updated dataset to a new Excel file
output_path = '/mnt/data/canada_data_cleaned.xlsx'
cleaned_data.to_excel(output_path, index=False)

# Display the Python code and link to download the cleaned file
output_path
