
#################################################################
import json
from pathlib import Path
import os
import sys
import time
from typing import List
import pandas as pd
from z3 import *
sys.path.append(os.path.join(Path(__file__).parent.parent, 'algorithms', 'metrics'))
from prescription import Prescription

start = time.time()


with open("data/stackoverflow/rules_greedy_all.json") as f:
    data = json.load(f)
rxCandidates = []

for rule in data:
    rxCandidates.append(Prescription(rule['condition'], rule['treatment'], set(rule['coverage']), set(rule['protected_coverage']), rule['utility'], rule['protected_utility']))
df = pd.read_csv("data/stackoverflow/so_countries_col_new_mini.csv")
idx_all = set(df.index)

start = time.time()
m = len(idx_all)
l = len(rxCandidates)
potential_idx = set.union(*[rx.covered_idx for rx in rxCandidates])
potential_idx_p = set.union(*[rx.covered_idx_protected for rx in rxCandidates])
potential_idx_u = potential_idx - potential_idx_p  

# Create an inverse mapping that maps tuple to rules it could be covered
t_domain = [[f"g_{j}" for j, rx in enumerate(rxCandidates) if i in rx.covered_idx] for i in potential_idx]
t: List = [
        EnumSort(name = f"t_{ti}", values =t_domain[i])
        for i, ti in enumerate(potential_idx)
    ]
print(time.time() - start)
print()
#################################################################