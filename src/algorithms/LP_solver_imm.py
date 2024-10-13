import cProfile
import functools
import json
import logging
from pathlib import Path
import pstats
from typing import Dict, List
from attr import dataclass
from z3 import *
import multiprocessing
import os
import sys
import pandas as pd
sys.path.append(os.path.join(Path(__file__).parent.parent, 'tools'))
sys.path.append(os.path.join(Path(__file__).parent, 'metrics'))

from StopWatch import StopWatch
from prescription import Prescription


def LP_solver(rxCandidates: List[Prescription], idx_all, idx_protected, cvrg_constr, fair_constr, l1=1, l2=100000):
    """
    Objective: max[]
    Constains on
    Solve the Set Cover Problem using Linear Programming.

    Args:

    Returns:
        list: A list of selected set names that satisfy the constraints and maximize the objective.
    """
    ctx = main_ctx()
    solver = Optimize()
    m = len(idx_all)
    l = len(rxCandidates)
    idx_unprotected = set(idx_all) - idx_protected 
    mp = len(idx_protected)
    mu = len(idx_unprotected)
    # g[j] => rule j is selected
    g: List[z3.z3.BoolRef]= [Bool(f"g{j}") for j in range (l)]
    w = [int(rxCandidates[j].utility) - l2 for j in range(l)]


    potential_idx = set.union(*[If(g[j], rxCandidates[j].covered_idx, set()) for j in range(l) ])
    potential_idx_p = set.union(*[rx.covered_idx_p for rx in rxCandidates])
    potential_idx_u = potential_idx - potential_idx_p  
    # Make potential indices ordered 
    potential_idx = list(potential_idx) 

    # Create an inverse mapping that maps tuple to rules it could be covered
    t_domain = [set([j for j, rx in enumerate(rxCandidates) if i in rx.covered_idx]) for i in potential_idx]
    
    t: List = [
        EnumSort(name = f"t_{ti}", values =['g_-1']+t_domain[i])
        for i, ti in enumerate(potential_idx)
    ]
    # Constraint 1;
    # For all i, t[i] -> g[t[i]]  
    solver.add([Implies(ti != 'g_-1', ti.translate()) for ti in t]) 

    solver.add()

    num_protected: ArithRef = Sum(tp)
    num_unprotected: ArithRef = Sum(tu)
    # Constraint 3;
    # Group coverage (if any)
    if cvrg_constr != None and 'group' in cvrg_constr['variant']:
        threshold = cvrg_constr['threshold'] 
        threshold_p = cvrg_constr['threshold_p'] 
        solver.add(num_protected + num_protected >= (threshold * m))
        solver.add(num_protected >= (threshold_p * mp))

    # Constraint 4;
    # Group fairness (if any)
    if fair_constr != None:
        threshold = fair_constr['threshold'] 
        if 'group_sp' in fair_constr['variant']:
            ttl_util_p = Sum([Sum([If(t[i][j], w[j], 0) for i in idx_protected]) for j in range(l)])
            num_u =  Sum([Sum(t[i]) for i in idx_unprotected])
            ttl_util_u = Sum([Sum([If(t[i][j], w[j], 0) for i in idx_unprotected]) for j in range(l)])
            solver.add(Abs(ttl_util_p * num_u  - ttl_util_u * num_p)  < threshold * num_p * num_u)
    # Maximize the sum of weights while penalizing size of the set
    solver.maximize(Sum([g[j] * w[j] for j in range(l)])) 
    with StopWatch("Solving"):
        # Check for satisfiability and retrieve the optimal solution
        if solver.check() == sat:
            model = solver.model()
            selected_sets = [rxCandidates[j] for j in range(l) if is_true(model[g[j]])]
            print(len(selected_sets))
            return selected_sets
        else:
            logging.warning("No solution was found!")
            return []
    


def main():
    with open("/Users/bcyl/FairPrescriptionRules/output/10-01/20:36/greedy/rules_greedy_all.json") as f:
        data = json.load(f)
    rxCandidates = []
    for rule in data:
        rxCandidates.append(Prescription(rule['condition'], rule['treatment'], set(rule['coverage']), set(rule['protected_coverage']), rule['utility'], rule['protected_utility']))
    df = pd.read_csv("/Users/bcyl/FairPrescriptionRules/data/stackoverflow/so_countries_col_new_mini.csv")
    df = df.drop(['Unnamed: 0'], axis=1, errors='ignore')
    idx_protected = set(df[df['RaceEthnicity'] != 'White or of European descent'].index)
    l = 10
    m = 100
    set_param('sat.lookahead_simplify', True) 
    set_param("parallel.enable", True)
    set_param('parallel.threads.max', 10000)
    
    cvrg_constr = {
        'variant': 'group',
        'threshold': 0.8,
        'threshold_p': 0.8,
    }
    with StopWatch(""):
        LP_solver(rxCandidates, set(df.index), idx_protected, cvrg_constr, None)
    return

main()
