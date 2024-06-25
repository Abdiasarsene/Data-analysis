import pandas as pd

# URL of the dataset
url = "https://raw.githubusercontent.com/OpenClassrooms-Student-Center/8063076-Initiez-vous-au-Machine-Learning/master/data/palmer_penguins_openclassrooms.csv"

# Load the dataset into a pandas dataframe
penguins_df = pd.read_csv(url)

# Display the first few rows of the dataframe
penguins_df.head(), penguins_df.columns

# 