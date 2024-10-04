import cProfile
import functools
import json
import logging
from pathlib import Path
import pstats
from typing import Dict, List
from attr import dataclass
from scipy import optimize
from z3 import *
import multiprocessing
import os
import sys
sys.path.append(os.path.join(Path(__file__).parent.parent, 'tools'))
sys.path.append(os.path.join(Path(__file__).parent, 'metrics'))

from StopWatch import StopWatch
from prescription import Prescription
set_param('sat.lookahead_simplify', True) 
set_param("parallel.enable", True)
set_param('parallel.threads.max', 10000)


def LP_solver(rxCandidates, idx_all, idx_protected, cvrg_constr, fair_constr, l1=1, l2=200000):
    """
    Objective: max[]
    Constains on
    Solve the Set Cover Problem using Linear Programming.

    Args:

    Returns:
        list: A list of selected set names that satisfy the constraints and maximize the objective.
    """
    # TODO add soft constraint


    # Sort Rx candidates so that low utility rules are selected first.
    # Assuming that each individual always choose the worse treatment
    rxCandidates.sort(reverse=True, key=lambda rx: rx.utility)
    solver = Optimize()
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
    # Scale the utilities by their protected/unprotected size

    # Maximize the sum of weights while penalizing size of the set
    objective = Sum([g[j] * w[j] for j in range(l)])

    solver.maximize(objective)
    solver.add(objective < 1000000)
    solver.add(objective > 0)
    
    # Constraint 1;
    # For all i, j: t[i][j] <= g[j] 
    # Equivalent to t[i][j] -> i in rx[j].covered and g[j]  
    for i in range(m):
        solver.add([Implies(t[i][j], i in rxCandidates[j].covered_idx and g[j]) for j in range(l)])

    # Constraint 2a;
    # For all j: g[j] <= Sum t[i][j]
    # Equivalent to g[j] => OR(t[i]) 
    solver.add([Implies(g[j], PbGe([(t[i][j], 1) for i in range(m)], 1)) for j in range(l)])     
    # Constraint 2b;
    # For all i: sum (t[i][j]) <= 1
    solver.add([PbLe([(tij, 1) for tij in t[i]], 1) for i in range(m)])  
    # Constraint 3;
    # Group coverage (if any)
    if cvrg_constr != None and 'group' in cvrg_constr['variant']:
        threshold = cvrg_constr['threshold'] 
        threshold_p = cvrg_constr['threshold_p'] 
        solver.add(PbGe([(Or([t[i][j] for j in range(l)]), 1) for i in idx_all], int(threshold * (mp + mu))))
        solver.add(PbGe([(Or([t[i][j] for j in range(l)]), 1) for i in idx_protected], int(threshold_p * mp)))

    # Constraint 4;
    # Group fairness (if any)
    if fair_constr != None:
        threshold = fair_constr['threshold'] 
        if 'group_sp' in fair_constr['variant']:
            num_p = Sum([Sum(t[i]) for i in idx_unprotected])
            ttl_util_p = Sum([Sum([If(t[i][j], w[j], 0) for i in idx_protected]) for j in range(l)])
            num_u =  Sum([Sum(t[i]) for i in idx_unprotected])
            ttl_util_u = Sum([Sum([If(t[i][j], w[j], 0) for i in idx_unprotected]) for j in range(l)])
            solver.add(Abs(ttl_util_p / num_p  - ttl_util_u * num_u)  < threshold)
            
    # Check for satisfiability and retrieve the optimal solution
    if solver.check() == sat:
        model = solver.model()
        selected_sets = [rxCandidates[j] for j in range(l) if is_true(model[g[j]])]
        print(len(selected_sets))
        return selected_sets
    else:
        logging.warning("No solution was found!")
        return []