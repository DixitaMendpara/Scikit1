import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv('headbrain.csv')
print(df.head())

df.columns = ['Head_size', 'Brain_weight']
model = smf.ols(formula='Head_size ~ Brain_weight',data=df).fit()
print(model.summary())