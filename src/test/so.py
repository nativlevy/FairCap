#!/usr/bin/env python
# coding: utf-8

# In[ ]
import pandas as pd
df = pd.read_csv("../../data/stackoverflow/so_countries_col_new.csv")
df = df.drop(['Unnamed: 0'], axis=1, errors='ignore')

idx_protec = set(df[df['RaceEthnicity'] != 'White or of European descent'].index)
df_g = df[df['YearsCoding']=="12-14 years"]
mask = df_g.index.isin(idx_protec)
# %%
newTreatment = {"Exercise": "I don't typically exercise",
            "Country": "United States",
            "Student": "Yes, full-time"
        }
grp = (df_g[newTreatment.keys()] != newTreatment.values()).all(axis=1)
sum(grp) / len(df)
print(len(df_g[grp == 1]))
keys = list(newTreatment.keys())
vals = list(newTreatment.values())
  
print(len(df_g))
treatable = (df_g[keys] != vals).any(axis=1)
valid = list(set(treatable.tolist()))
# no tuples in treatment group
size = len(df_g[treatable == 1])
print(size)
# %%
