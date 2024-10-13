#!/usr/bin/env python
# coding: utf-8

# In[ ]
import array
import timeit
from numpy import astype
import pandas as pd
from apyori import apriori
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
import logging
import os




def removeHeader(df_org, name):
    """
    Remove header from the dataframe and process it for transaction analysis.

    This function creates a new column for each original column, combining the column name and its value.
    It then saves the processed dataframe to a CSV file and reloads it without headers.

    Args:
        df_org (pd.DataFrame): The original dataframe.
        name (str): The name of the output CSV file.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The processed dataframe.
            - int: Number of rows in the processed dataframe.
            - int: Number of columns in the processed dataframe.
    """
    logging.debug(f"Removing header and processing dataframe for {name}")
    df = df_org.copy(deep = True)
    columns = df.columns
    for col in columns:
        df[col] = df.apply(lambda row: addColName(row, col), axis=1)

    # Save to CSV without header and index
    df.to_csv(name, header=None, index=False)

    # Read the CSV back
    df = pd.read_csv(name, header=None)
    rows = len(df)
    columns = len(df.columns)
    logging.debug(f"Size of df: {rows} rows, {columns} columns")
    return df, rows, columns


def addColName(row, col):
    """
    Concatenate column name and value for a given row and column.

    Args:
        row (pd.Series): A row from the dataframe.
        col (str): The column name.

    Returns:
        str: A string in the format "column_name = value".
    """
    val = row[col]
    return col +" = "+str(val)

t = 0.1

# In[ ]:
df = pd.read_csv("/Users/bcyl/FairPrescriptionRules/data/stackoverflow/so_countries_col_new_mini.csv")
df = df.drop(['Unnamed: 0'], axis=1, errors='ignore')

protected_group = set(df[df['RaceEthnicity'] != 'White or of European descent'].index)
df_og = df
atts = [
        'Gender', 'SexualOrientation', 'EducationParents', 'RaceEthnicity',
        'Age', 'YearsCoding', 'Dependents',
]
df = df[atts]
df, rows, columns = removeHeader(df, 'Temp.csv')

# In[]:
df = pd.read_csv("/Users/bcyl/FairPrescriptionRules/data/stackoverflow/so_countries_col_new_mini.csv")
df = df.drop(['Unnamed: 0'], axis=1, errors='ignore')


start_time = timeit.default_timer()
# code you want to evaluate
atts = [
        'Gender', 'SexualOrientation', 'EducationParents', 'RaceEthnicity',
        'Age', 'YearsCoding', 'Dependents',
]
df = df[atts]
df, rows, columns = removeHeader(df, 'Temp.csv')
te = TransactionEncoder()
records = []
for i in range(0, rows):
    records.append([str(df.values[i, j]) for j in range(0, columns)])

logging.debug(f"Number of records: {len(records)}")

te = TransactionEncoder()

te_ary = te.fit_transform(records)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets1 = apriori(df, min_support=t, use_colnames=True)

rules1 = []
for index, row in frequent_itemsets1.iterrows():
    parts = set(row["itemsets"])
    temp = {}
    for part in parts:
        part = part.split("=")
        temp[part[0].strip()] = part[1].strip()
    rules1.append(temp)
elapsed1 = timeit.default_timer() - start_time

# In[ ]:

def underscore_combiner(col_name, entry):
    return f"{col_name} = {entry}"

# In[]:

from sklearn.preprocessing import OneHotEncoder
df = pd.read_csv("/Users/bcyl/FairPrescriptionRules/data/stackoverflow/so_countries_col_new_mini.csv")
df = df.drop(['Unnamed: 0'], axis=1, errors='ignore')
df = df[atts]
def entry_with_col_name(col_name, entry):
    """Prefix an entry with it's, combined with ' = '
    e.g:
        -------------------------
        | Age                   |
        | '18 - 24 years old'   | 
        -------------------------
        becomes 
        -----------------------------
        | Age                       |
        | 'Age==18 - 24 years old'  | 
        -----------------------------
    
    """
    return f"{col_name}=={entry}"
enc = OneHotEncoder(handle_unknown='ignore', feature_name_combiner=entry_with_col_name, sparse_output=False)
enc.set_output(transform = 'pandas')
df = enc.fit_transform(df)
df = df.reindex(sorted(df.columns), axis=1)
frequent_itemsets2 = apriori(df, min_support=t, use_colnames=True)
def set_to_dict(s):
    """Transport a string to to a tuple in the dictionary

    Args:
        s (list[str]): list of attributes in the format:
            x_y

    Returns:
        _type_: _description_
    """
    _rule = {}
    for i in s:
        k, v = i.split('==')
        _rule[k] = v
    return _rule
rules = list(map(set_to_dict, frequent_itemsets2['itemsets'])) 

# %%
frequent_itemsets1.sort_values(by=['support'])



# %%

grouping_patterns = rules
def apply_pattern(pattern):
    mask = pd.Series(True, index=df.index)
    for col, val in pattern.items():
        mask &= df[col] == val
    return frozenset(df.index[mask])

# Create a dictionary to store patterns by their coverage
coverage_dict = {}
for pattern in grouping_patterns:
    coverage = apply_pattern(pattern)
    if coverage in coverage_dict:
        if len(pattern) < len(coverage_dict[coverage]):
            coverage_dict[coverage] = pattern
    else:
        coverage_dict[coverage] = pattern

filtered_patterns = list(coverage_dict.values())

# %%
import pygraphviz as pgv
DAG = pgv.AGraph('/Users/bcyl/FairPrescriptionRules/data/stackoverflow/so.dot', directed=True)
DAG.edges()
V, E = DAG.nodes(), list(map(lambda a: "%s -> %s" % a, DAG.edges()))
[*V, *E]

# %%
import json
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("/Users/bcyl/FairPrescriptionRules/data/stackoverflow/so_countries_col_new_500.csv")
asProtected = True
attrP = 'GDP'
valP = 'Low'
df_protec = None
if asProtected:
    df_protec = df[(df[attrP] == valP)]
else:
    df_protec = df[(df[attrP] != valP)]
idx_p = set(df_protec.index)

df_unprotec = df.loc[df.index.difference(idx_p)]

# %%
plt.hist( df[df['GDP'] == 'High']['ConvertedSalary'], range=[0,200000])

plt.figure()
plt.hist(df[df['GDP'] == 'Medium']['ConvertedSalary'], range=[0,200000])
plt.figure()

plt.hist(df[df['GDP'] == 'Low']['ConvertedSalary'], range=[0,200000])

print(df.groupby(['GDP'])['ConvertedSalary'].mean())
print(set(df[df['GDP'] == 'Low']['Country']))
# %%
print(len(df[df['Country'] == 'United States'])/len(df))
# %%
for i in df.columns:

    plt.figure()
    plt.xlabel(i)
    plt.scatter(df[i], df['ConvertedSalary'])
    plt.plot()
# %%
for i in df.columns:
    plt.figure()
    plt.xlabel(i)

    plt.plot(df[df['GDP'] == 'Low'].groupby([i])['ConvertedSalary'].mean())



    plt.plot(df[df['GDP'] != 'Low'].groupby([i])['ConvertedSalary'].mean())
    plt.xticks(rotation='vertical')

# %%
plt.hist(df[df['Student'] != 'No']['ConvertedSalary'])
# %%
for i in [    "Exercise",
    "HoursComputer",
    "FormalEducation",
    "YearsCoding"]:
    print(f"\"{i}\":{list(set(df[i]))}")

# %%
attrOrdinal =  {
    "Exercise": [
      "I don't typically exercise",
      "Daily or almost every day",
      "1 - 2 times per week",
      "3 - 4 times per week",
      "UNKNOWN"
    ],
    "HoursComputer": [
      "Less than 1 hour",
      "1 - 4 hours",
      "5 - 8 hours",
      "9 - 12 hours",
      "Over 12 hours",
      "UNKNOWN"
    ],
    "FormalEducation": [
      "I never completed any formal education",
      "Primary/elementary school",
      "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)",
      "Associate degree",
      "Some college/university study without earning a degree",
      "Bachelor’s degree (BA, BS, B.Eng., etc.)",
      "Master’s degree (MA, MS, M.Eng., MBA, etc.)",
      "Other doctoral degree (Ph.D, Ed.D., etc.)",
      "Professional degree (JD, MD, etc.)"
    ],
    "YearsCoding": [
      "0-2 years",
      "3-5 years",
      "6-8 years",
      "9-11 years",
      "12-14 years",
      "15-17 years",
      "18-20 years",
      "21-23 years",
      "24-26 years",
      "27-29 years",
      "30 or more years"
    ]
  }
treat = {
    'YearsCoding': '12-14 years'
}
df[df[treat.keys()] > treat.values()]['YearsCoding']
# %%
cond = {
            "RaceEthnicity": "White or of European descent",
            "Age": "25 - 34 years old",
            "SexualOrientation": "Straight or heterosexual",
            "EducationParents": "Bachelor\u2019s degree (BA, BS, B.Eng., etc.)"
        }
df_g = df[(df[cond.keys()] == cond.values()).all(axis=1)]

sum(df_g['GDP'] == 'Low')
# %%

df_unprotec
# %%
