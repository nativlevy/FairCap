#!/usr/bin/env python
# coding: utf-8

# In[ ]
import pandas as pd
df = pd.read_csv("../../data/stackoverflow/so_countries_col_new.csv")
df = df.drop(['Unnamed: 0'], axis=1, errors='ignore')

idx_protec = set(df[df['RaceEthnicity'] != 'White or of European descent'].index)
grouping = {
            "YearsCoding": "3-5 years",
            "SexualOrientation": "Straight or heterosexual"
        } 

df_g = df[(df[grouping.keys()] != grouping.values()).all(axis=1)]
mask = df_g.index.isin(idx_protec)
# %%
treatment = {
            "HoursComputer": "5 - 8 hours",
            "Continent": "North America"
        }
control =(df_g[treatment.keys()] == treatment.values()).all(axis=1)
 
treated = (df_g[treatment.keys()] != treatment.values()).any(axis=1)
sum(treated) / len(df)
print(len(df_g[treated == 1]))
keys = list(treatment.keys())
vals = list(treatment.values())
  
print(len(df_g))
treatable = (df_g[keys] != vals).any(axis=1)
valid = list(set(treatable.tolist()))
# no tuples in treatment group
size = len(df_g[treatable == 1])
print(size)
# %%

df_g[control]['ConvertedSalary'].mean()
df_g[treated]['ConvertedSalary'].mean()

# %%
