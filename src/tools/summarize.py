# %%
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Ouptut se
plt.style.use('ggplot')

variants = ['greedy_no_constraint',
        'greedy+group_coverage',
        'greedy+rule_coverage',
        'greedy+group_sp',
        'greedy+individual_sp',
        'greedy+group_coverage+group_sp',
        'greedy+rule_coverage+group_sp',
        'greedy+group_coverage+individual_sp',
        'greedy+rule_coverage+individual_sp',
]
# %%
def expUtitVsK(path):
    variants = os.listdir(path)
    variants = [v for v in variants if 'greedy' in v]

    i = 0
    plt.figure(figsize=(20,20))
    plt.subplots_adjust(hspace=.5)
    for v in variants:
        i+=1
        plt.subplot(3,3,i)
        plt.title(v)
        plt.xlabel('k')
        plt.ylabel('ExpUtil')
        plt.legend()
        try:
            df = pd.read_csv(f"{path}/{v}/experiment_results_greedy.csv")

            plt.plot(df['k'], df['expected_utility'], label='Expected')
            plt.plot(df['k'], df['unprotected_expected_utility'], label='Unprotected Expected')
            plt.plot(df['k'], df['protected_expected_utility'], label='Protected Expected')
            plt.legend(loc='best')
        except:
            pass

    plt.plot()
# %%
def execTimeBreakDown(path):
    variants = os.listdir(path)
    variants = [v for v in variants if 'greedy' in v]
    i = 0
    plt.figure(figsize=(20,20))
    breakdown = []
    for v in variants:
        with open (f"{path}/{v}/stdout.log") as f:
            matches = re.findall(r"([\d:,.]+) seconds?", f.read())
        breakdown.append(matches)
    stages = ['Group Mining', 'Treatment Mining', 'Rule Selection']
    
    df = pd.DataFrame(breakdown, index = variants, columns=stages, dtype='float')
    ttl_time = np.sum(df, axis=1)
    sorted_idx = np.argsort(ttl_time)
    df=df.iloc[sorted_idx]

    base = pd.Series([0.0 for i in range(len(variants))], index=df.index)
    for s in stages:
        p = plt.barh(df.index, df[s], label=s, left=base)
        base += df[s]
    plt.legend(loc='best')

    plt.show()

# %%
def print_table(path):

    i = 0
    fields = [
    ' # rules ',
    ' coverage ',
    ' coverage pro ',
    ' exp utility ',
    ' exp utility non-pro',
    'exp utility pro ',
    'unfairness']
    table=[]
    for v in variants:
        i+=1
        row = []

    
        df = pd.read_csv(f"{path}/{v}/experiment_results_greedy.csv")

        k = min(20, max(df['k']))
      
        rec = (df.loc[k-1])

        row.append(k)
        row.append(rec['coverage_rate'])
        row.append(rec['protected_coverage_rate'])
        row.append(round(float(rec['expected_utility']),2))
        row.append(round(float(rec['unprotected_expected_utility']),2))
        row.append(round(float(rec['protected_expected_utility']),2))
        row.append(round(float(rec['unprotected_expected_utility'] - rec['protected_expected_utility']),2))
        row = [str(i).replace("%", "\\%") for i in row]
        table.append('& '.join(row))    



    t = table
    print(f"""\midrule \\\\
No constraints  & {t[0]} \\\\
Group coverage  &{t[1]} \\\\
Rule coverage  & {t[2]}\\\\
Group fairness  & {t[3]} \\\\
Individual fairness   & {t[4]} \\\\
Group coverage, Group fairness  & {t[5]} \\\\
Rule coverage, Group fairness  & {t[6]}\\\\
Group coverage, Individual fairness  & {t[7]}\\\\
Rule coverage, Individual fairness  &{t[8]} \\\\
        """)
path = "/Users/bcyl/FairPrescriptionRules/output/GDP/so_full"

print_table(path)

# %%

path = "/Users/bcyl/FairPrescriptionRules/output/10-15/17:01"
expUtitVsK(path)
execTimeBreakDown(path)


# %%
