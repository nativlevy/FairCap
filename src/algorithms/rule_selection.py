
# pickleable for multiprocessing 
from functools import partial
import logging
from math import inf
import multiprocessing
from multiprocessing import pool, Manager
import os
import time
from typing import List
from uu import Error

from prescription import Prescription, PrescriptionList


def findBestRxListExperimental(ns, newRx: Prescription):
    """Experimental
    """
    selectedRx: PrescriptionList = ns.selectedRx 
    cvrg_constr = ns.cvrg_constr
    fair_constr = ns.fair_constr
    newSelectedRx = selectedRx.addRx(newRx) 
    newExpUtil = newSelectedRx.expected_utility
    newExpUtilU = newSelectedRx.expected_utility_u
    newExpUtilP = newSelectedRx.expected_utility_p
    # If there is no group fairness, the benefit is simply the expected utility
    score = newExpUtil

    if cvrg_constr == None or 'group' not in cvrg_constr['variant']:
        # When coverage is met, put more weights on utility
        score = score * (len(newSelectedRx.idx_all) / len(newSelectedRx.covered_idx))

    if fair_constr != None and 'group_sp' in fair_constr['variant']:
        threshold = fair_constr['threshold']
        if abs(newExpUtilU - newExpUtilP) > threshold:
            score -= (abs(newExpUtilU - newExpUtilP) - threshold)

    if fair_constr != None and 'group_bgl' in fair_constr['variant']:
        threshold = fair_constr['threshold']
        if newExpUtilP < threshold:
            score -= (threshold - newExpUtilP)
 
    if score > ns.best_score:
        ns.best_score = score        
        ns.best_newRx = newRx
        ns.best_exp_util = newExpUtil
        ns.best_newRxList = newSelectedRx

# Strictly prescribed in manuscript
def findBestRxList(ns, newRx: Prescription):
    selectedRx: PrescriptionList = ns.selectedRx 
    cvrg_constr = ns.cvrg_constr
    fair_constr = ns.fair_constr
    newSelectedRx = selectedRx.addRx(newRx) 
    newExpUtil = newSelectedRx.expected_utility
    newExpUtilU = newSelectedRx.expected_utility_u
    newExpUtilP = newSelectedRx.expected_utility_p
    # If there is no group fairness, the benefit is simply the expected utility
    score = newExpUtil

    if fair_constr != None and 'group' in fair_constr['variant']:
        benefit = newExpUtilU
        # Unfairness penalize the score
        if 'group_sp' in fair_constr['variant']:
            threshold = fair_constr['threshold']
            if newExpUtilU - newExpUtilP > threshold:
                benefit = newExpUtil / newExpUtilU - newExpUtilP
                
        elif'group_bgl' in fair_constr['variant']:
            threshold = fair_constr['threshold']
            if newExpUtilP < threshold:
                benefit = newExpUtilU / (threshold - newExpUtilP)
        score += benefit
    if cvrg_constr != None and 'group' in cvrg_constr['variant']:
        # TODO? Put weights on protected coverage as well?
        score *= min(selectedRx.getCoverageRate(), selectedRx.getProtectedCoverageRate())
    if score > ns.best_score:
        ns.best_score = score
        ns.best_exp_util = newExpUtil 
        ns.best_newRx = newRx
        ns.best_newRxList = newSelectedRx

def k_selection(k, idx_all, idx_p, rxCandidates: List[Prescription], cvrg_constr, fair_constr):
    nCPU = os.cpu_count()
    unselectedRx = {rx.name: rx for rx in rxCandidates}
    cvrg_met = cvrg_constr == None or 'group' not in cvrg_constr['variant'] 
    fair_met = fair_constr == None or 'group' not in fair_constr['variant'] 
    mgr = Manager()
    ns = mgr.Namespace()
    ns.cvrg_constr = cvrg_constr
    ns.fair_constr = fair_constr
    ns.selectedRx = PrescriptionList([], idx_all, idx_p) 
    kResults = []
    prev_best_score = float("-inf") 

    start_time = time.time() 

    for j in range(k):
        # Reset best rx and best score 
        ns.best_score = float("-inf")
        ns.best_newRx = None
        ns.best_newRxList = None
        with multiprocessing.Pool(processes=nCPU-1) as pool:
            pool.map(partial(findBestRxList, ns), unselectedRx.values())
        
        if ns.best_newRx == None:
            logging.error("No suitable rules to select")
            return ns.selectedRx, kResults
        del unselectedRx[ns.best_newRx.name]
            
        # If constraints are met and new rule does not increase score, then return the previous best solution  
        if cvrg_met and fair_met and ns.best_score <= prev_best_score:
            return ns.selectedRx, kResults 
        
        new_fair_met = ns.best_newRxList.isFairnessMet(fair_constr)
        cvrg_met = cvrg_met or ns.best_newRxList.isCoverageMet(cvrg_constr)
        
        # If fairness was met but is violated in new selection, stop selection  
        if cvrg_met and fair_met and not new_fair_met:
            return ns.selectedRx, kResults 
        fair_met = new_fair_met 
        # Once coverage constraint is met, we can focus only on expected
        # utility and fairness (if constrained)
    
        prev_best_score = ns.best_score
        ns.selectedRx = ns.best_newRxList
        kResults.append ({
            'k': j + 1,
            'expected_utility': ns.selectedRx.getExpectedUtility(),  
            'unprotected_expected_utility': ns.selectedRx.getUnrotectedExpectedUtility(), 
            'protected_expected_utility': ns.selectedRx.getProtectedExpectedUtility(),
            'coverage_rate': f"{ns.selectedRx.getCoverageRate() * 100}%",
            'protected_coverage_rate':  f"{ns.selectedRx.getProtectedCoverageRate() * 100}%",
            'execution_time': time.time() - start_time,
            'fairness_met': fair_met,
            'coverage_met': cvrg_met
        })
        
    return ns.selectedRx, kResults
        

