from heapq import nlargest
import logging
import os
import sys
from pathlib import Path
import pandas as pd
from typing import List, Set, Dict, Tuple

import time
import csv
import json
import statistics
import concurrent


SRC_PATH = Path(__file__).parent.parent 
sys.path.append(os.path.join(SRC_PATH, 'tools'))
sys.path.append(os.path.join(SRC_PATH, 'algorithms'))
sys.path.append(os.path.join(SRC_PATH, 'algorithms', 'metrics'))
from group_mining import getConstrGroups
from treatment_mining import getTreatmentForAllGroups
from rule_selection import k_selection
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
from load_data import load_data, load_rules
from prescription import Prescription, PrescriptionList

# from utility.logging_util import init_logger

sys.path.append(os.path.join(Path(__file__).parent, 'common'))
from consts import APRIORI, MIX_K, MAX_K, DATA_PATH, PROJECT_PATH, unprotected_coverage_threshold, protected_coverage_threshold, fairness_threshold  # NOQA



def main_cmd(data_config_path, output_path):
    with open(data_config_path) as json_file:
        data_config = json.load(json_file)   
    os.makedirs(output_path, exist_ok=True)
    main(data_config, output_path)

# TODO unwind me


def main(config, output_path):
    """
    Main function to run the greedy fair prescription rules algorithm for different values of k.
    """
    # ------------------------ PARSING CONFIG BEGINS  -------------------------

    """
        attrI := Immutable/unactionable attributes
        attrM := Mutable/actionable attributes
        attrP := Protected attributes
        valP  := Values of protected attributes
        tgt   := Target outcome
    """
    dataset_path = config.get('_dataset_path')
    datatable_path = config.get('_datatable_path')
    dag_path = config.get('_dag_path')
    attrI = config.get('_immutable_attributes')
    attrM = config.get('_mutable_attributes')
    attrP = config.get('_protected_attributes')
    attrOrdinal = config.get('_ordinal_attributes') 
    # TODO extend to support multiple protected value
    valP, asProtected = config.get('_protected_values')
    tgtO =  config.get('_target_outcome')
    cvrg_constr = config.get('_coverage_constraint', None)
    fair_constr = config.get('_fairness_constraint', None)

    MIN_K, MAX_K = config.get('_k', [4,4])
    # Remove protected attributes from immutable attributes
    attrI.remove(attrP) 

    # ------------------------- PARSING CONFIG ENDS  -------------------------
    # ------------------------ DATASET SETUP BEGINS -------------------------- 
    df, DAG_str =load_data(os.path.join(DATA_PATH, dataset_path, datatable_path), os.path.join(DATA_PATH, dataset_path, dag_path))
    df['TempTreatment'] = 0
    # Define protected group
    
    df_protec = None
    if asProtected:
        df_protec = df[(df[attrP] == valP)]
    else:
        df_protec = df[(df[attrP] != valP)]
    idx_p: Set = set(df_protec.index)
    logging.debug(
        f"Protected group size: {len(idx_p)} out of {len(df)} total")
    # ------------------------ DATASET SETUP ENDS ----------------------------

    # -------------------------- Group mining -----------------------------
    if True:
        start_time = time.time()
        # Step 1. Grouping pattern mining
        
        groupPatterns = getConstrGroups(df, idx_p, attrI, min_sup=APRIORI, cvrg_constr=cvrg_constr)

        exec_time1 = time.time() - start_time 
        print(f"Elapsed time for group mining: {exec_time1} seconds. {len(groupPatterns)} groups are found")

    # ------------------------ Treatment mining -----------------------------
    if True:
        start_time = time.time()
        # Step 2. Treatment mining using greedy
        # Get treatments for each grouping pattern
        logging.debug("Step2: Getting candidate treatments for each grouping pattern")
        rxCandidates:list[Prescription] = getTreatmentForAllGroups(DAG_str, df, idx_p, groupPatterns, attrOrdinal, tgtO, attrM, fair_constr=fair_constr)
        exec_time2 = time.time() - start_time 
        print(f"Elapsed time for treatment mining: {exec_time2} seconds. {len(rxCandidates)} rules are found")
        # Save all rules found so far
        with open(os.path.join(output_path, 'all_mined_rules.json'), 'w+') as f:
            json.dump([{
                'condition': rx.condition,
                'treatment': rx.treatment,
                'utility': rx.utility, 
                'protected_utility': rx.getProtectedUtility(),
                'unprotected_utility': rx.getUnprotectedUtility(),
                'coverage_rate': round(rx.getCoverage()/len(df) * 100, 2),
                'protected_coverage_rate': round(rx.getProtectedCoverage()/len(idx_p) *100, 2)
            } for rx in rxCandidates], f, indent=4)
        # rxCandidates = LP_solver_k(rxCandidates, set(df.index), idx_p, cvrg_constr, fair_constr, 10)
        
    # ------------------------ Rule selections -----------------------------
    # rxCandidates = load_rules(f"{DATA_PATH}/stackoverflow/mined_rules.json")
    # exec_time1 = 0
    # exec_time2 = 0
    # TODO remove debug

    start_time = time.time()
    rxSelected, kResults = k_selection(min(len(rxCandidates), 50), set(df.index), idx_p, rxCandidates, cvrg_constr, fair_constr) 
    exec_time3 = time.time() - start_time
    print(f"Elapsed time for Selection: {exec_time3} seconds")
    with open(os.path.join(output_path, 'experiment_results_greedy.csv'), 'w+', newline='') as csvfile:
        fieldnames = ['k', 'expected_utility', 'unprotected_expected_utility', 'protected_expected_utility', 'coverage_rate',
                      'protected_coverage_rate', 'execution_time',]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in kResults:
            writer.writerow({
                'k': result["k"],
                'expected_utility': result['expected_utility'],   
                'unprotected_expected_utility': result['unprotected_expected_utility'],
 
                'protected_expected_utility': result['protected_expected_utility'],
                'coverage_rate': result['coverage_rate'],
                'protected_coverage_rate': result['protected_coverage_rate'],
                                'execution_time': exec_time1 + exec_time2 + result['execution_time']

            })

    # Convert selected_rules to a JSON string
    with open(os.path.join(output_path, 'selected_rules.json'), 'w+') as f:
        json.dump([{
        'condition': rx.getGroup(),
        'treatment': rx.getTreatment(),
        'utility': rx.getUtility(),
        'protected_utility': rx.getProtectedUtility(),
        'unprotected_utility': rx.getUnprotectedUtility(),
        'coverage_rate': round(rx.getCoverage()/len(df) * 100, 2),
        'protected_coverage_rate': round(rx.getProtectedCoverage()/len(idx_p) *100, 2)
    } for rx in rxSelected.rules], f, indent=4)
    logging.debug("Results written to experiment_results_greedy.csv")
   


if __name__ == "__main__":
    main_cmd(sys.argv[1], sys.argv[2])
