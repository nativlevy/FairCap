import cProfile
from copy import deepcopy
import functools
import json
import logging
from multiprocessing import process
from multiprocessing.pool import ThreadPool
from pathlib import Path
import pstats
from typing import Dict, List
from attr import dataclass
from z3 import *
import multiprocessing
import os
import sys
sys.path.append(os.path.join(Path(__file__).parent.parent, 'tools'))
sys.path.append(os.path.join(Path(__file__).parent, 'metrics'))

from StopWatch import StopWatch
from prescription import Prescription

def util_obj():
    pass

def size_objc(isSelected, k):
    return Sum(list(isSelected)) <= k


    

def group_fairness_constr(candidateRx, idx_all, idx_protec, variant, threshold) -> z3.z3.BoolRef:
    # return a Z3 constraint
    if 'sp' in variant or 'bgl' in variant:
        partialMinUtil = functools.partial(minUtil, candidateRx)
        minUtils = {}
        with multiprocessing.Pool() as pool:
            minUtils = dict(zip(idx_all, pool.map(partialMinUtil, idx_all)))
        minProtectedUtils = {idx: minUtil for idx, minUtil in minUtils.items() if idx in idx_protec}
        if 'sp' in variant:
            return  
        
    else:   
        raise ArgumentError(f"Unsupported fairness variant: {variant}")
    

 
def minUtil(rxSet, idx):
    applicableRxSet = list(filter(lambda rx: idx in rx.covered, rxSet))
    leastEffectiveRx = min(applicableRxSet, key=lambda rx: rx.utility) 
    return leastEffectiveRx.getUtility() 
    
# def LP_solver_w_size_constr(g, t, lo, hi, ctx):

#     # Parallel creation of z3 object
#     condition = And(lo <=Sum(g), Sum(g) < hi, ctx)
#     # Parallel solving
#     solver = Solver(ctx=ctx)
#     solver.add(condition)
#     if solver.check() == sat:
#         model = solver.model()
#         return model
#     return None


def LP_solver_w_size_constr(rxCandidates: List[Prescription], idx_all, idx_protected, cvrg_constr, fair_constr, l1=1, l2=150000, rule_num_range = None):
    """
    Objective: max[]
    Constains on
    Solve the Set Cover Problem using Linear Programming.

    Args:

    Returns:
        list: A list of selected set names that satisfy the constraints and maximize the objective.
    """
    if rule_num_range == None:
        rule_num_range == [1, len(rxCandidates)]
    lo, hi = rule_num_range 

    solver = Optimize()
    solver.add(And(lo <= Sum(g), Sum(g) < hi))
    m = len(idx_all)
    l = len(rxCandidates)
    idx_unprotected = set(idx_all) - idx_protected 
    mp = len(idx_protected)
    mu = len(idx_unprotected)
    # g[j] => rule j is selected
    g: List[z3.z3.BoolRef]= [Bool(f"g{j}") for j in range (l)]
    # t[i][j] => t[i] is covered by and takes rule j as Rx
    t: List[List[z3.z3.BoolRef]]= [[Bool(f"t{i}_{j}") for j in range(l)] for i in range(m)]

    w = [rxCandidates[j].utility - l2 for j in range(l)]


    # Constraint 1;
    # For all i, j: t[i][j] <= g[j] 
    # Equivalent to 
    with StopWatch(["c 1"]):
        for i in range(m):
            solver.add([Implies(t[i][j], i in rxCandidates[j].covered_idx and g[j]) for j in range(l)])

    with StopWatch(["c 2a"]):

        # Constraint 2a;
        # For all j: g[j] <= Sum t[i][j]
        # Equivalent to g[j] => OR(t[i]) 
        solver.add([Implies(g[j], Or([i in rxCandidates[j].covered_idx and t[i][j] for i in range(m)])) for j in range(l)]) 
    with StopWatch(["c 2b"]):
        # Constraint 2b;
        # For all i: sum (t[i][j]) <= 1
        solver.add([Sum([t[i][j] for j in range(l)]) <= 1 for i in range(m)])  
    with StopWatch(["c 3"]):

        # Constraint 3;
        # Group coverage (if any)
        if cvrg_constr != None and 'group' in cvrg_constr['variant']:
            threshold = cvrg_constr['threshold'] 
            threshold_p = cvrg_constr['threshold_p'] 
            solver.add(Sum([Or([t[i][j] for j in range(l)]) for i in range(m)]) < threshold * m)
            solver.add(Sum([Or([t[i][j] for j in range(l)]) for i in idx_protected]) < threshold_p * m)
    with StopWatch(["c 4"]):

        # Constraint 4;
        # Group fairness (if any)
        if fair_constr != None:
            threshold = fair_constr['threshold'] 
            if 'group_sp' in fair_constr['variant']:
                exp_util_p = Sum([Sum([t[i][j] * w[j] for i in idx_protected]) for j in range(l)]) / Sum([Sum(t[i]) for i in idx_protected])
                exp_util_u = Sum([Sum([t[i][j] * w[j] for i in idx_unprotected]) for j in range(l)]) / Sum([Sum(t[i]) for i in idx_unprotected])
                solver.add(Abs(exp_util_p - exp_util_u)  < threshold)
    with StopWatch(["add solv"]):
        # Maximize the sum of weights while penalizing size of the set
        solver.maximize(Sum([g[j] * w[j] for j in range(l)]))
    with StopWatch(["solving "]):
        if solver.check() == sat:
            model = solver.model()
            selected_sets = [rxCandidates[j] for j in range(l) if is_true(model[g[j]])]
            exp_util = sum([g[j] * w[j] for j in range(l)])
        return exp_util, selected_sets
    return [0, None] 

def LP_solver(rxCandidates: List[Prescription], idx_all, idx_protected, cvrg_constr, fair_constr, l1=1, l2=150000):
    bestObjc = -1
    bestRxList = None
    step = len(rxCandidates) // (os.cpu_count() - 1) 
    ranges = [[i, i+step] for i in range(0, len(rxCandidates), step)]
    with multiprocessing.Pool(processes=os.cpu_count() - 1) as pool:
        results = [pool.apply_async(LP_solver_w_size_constr, args=(rxCandidates, idx_all, idx_protected, cvrg_constr, fair_constr, l1, l2, _range)) for _range in ranges]
    # Check for satisfiability and retrieve the optimal solution
    for r in results:
        objc, rxList = r.get()
        if objc > bestObjc:
            bestRxList = rxList
            bestObjc = objc 
    if bestObjc > 0: 
        return bestRxList
    else:
        logging.warning("No solution was found!")
        return []
   

import pandas as pd

def main():
    with open("data/stackoverflow/rules_greedy_all.json") as f:
        data = json.load(f)
    rxCandidates = []
    for rule in data:
        rxCandidates.append(Prescription(rule['condition'], rule['treatment'], set(rule['coverage']), set(rule['protected_coverage']), rule['utility'], rule['protected_utility']))
    df = pd.read_csv("data/stackoverflow/so_countries_col_new_500.csv")
    df = df.drop(['Unnamed: 0'], axis=1, errors='ignore')
    idx_protected = set(df[df['RaceEthnicity'] != 'White or of European descent'].index)
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