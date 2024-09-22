import pandas as pd
import missingno as msno
datacreated=pd.read_excel(r'c:\Users\HP ELITEBOOK 840 G6\Downloads\secteurs_modified.xlsx')
dataco=datacreated.head(500)
dataco.to_excel('dataco.xlsx', index=False)