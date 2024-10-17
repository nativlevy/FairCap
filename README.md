# Greedy Fair Prescription Rules Algorithm

This project implements a greedy algorithm for generating fair prescription rules. It aims to balance utility and fairness in decision-making processes, particularly focusing on protected groups.

# Table of Contents
1. [Overview](#overview) 
2. [Setup](#setup)   
3. [Running Experiments](#experiments)  
   3.a [Data configuration]()   
   3.b [Experiment and variants configuration]()
6. [Replication](#experiments)

## Overview  <a name="overview"></a> 

In this project, we implement 3-step algorithms that generate prescriptions(rules) to increase/decrease the value of an attribute while protecting a specified group. The algorithm can be broken down into 3 steps: group mining, treatment mining and rule selection. The details can be found in the [paper]()

## Setup <a name="setup"></a>
1. Clone this repository:
   ```
   git clone <repo_url>
   cd FairPrescriptionRules
   ```
You can run the algorithm either locally or **remotely(recommended)**
2. Local Setup (Skip to Remote setup)
### Environment
Linux:
```
sudo apt-get update
sudo apt-get install virtualenv
sudo apt-get install graphviz-dev
virtualenv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

Macintosh:
```
pip3 install virtualenv
brew install graphviz
virtualenv venv
source ./venv/bin/activate
pip install -r requirements.txt
```
### After installing dependencies. Run the sanity test to verify the result
```
cd reproducibility
sh local_sanity_check.sh
```

Expected output:
```
coverage constr: {'variant': 'rule', 'threshold': 0.8, 'threshold_p': 0.8}
fairness constr: None
Elapsed time for group mining: 0.0587611198425293 seconds. 3 groups are found
Elapsed time for treatment mining: 10.959924936294556 seconds. 3 rules are found
Elapsed time for Selection: 0.38343214988708496 seconds
```



## Installation <a name="installation"></a>



2. Install Graphviz:
   ```
   ```

3. Set up environment variables:
   ```
   export CFLAGS="-I$(brew --prefix graphviz)/include"
   export LDFLAGS="-L$(brew --prefix graphviz)/lib"
   ```

4. Install pygraphviz:
   ```
   pip install pygraphviz
   ```

5. Install other required packages:
   ```
   pip install -r requirements.txt
   ```

# Data

The datasets we use for evaluations are German credit and StackOverflow. They can be found in the `data` directory. 

## Running the Algorithms


## Greedy Algorithm Overview

1. Generate grouping patterns using the Apriori algorithm.
2. For each grouping pattern, find the best treatment that maximizes fairness and effectiveness.
3. Create Rule objects for each group-treatment pair, calculating utility and protected utility.
4. Initialize empty solution set and coverage tracking.
5. While the number of rules is less than the maximum allowed:
   - For each candidate rule, calculate a score based on:
      - The rule's utility
      - Coverage of both protected and unprotected groups
      - Fairness (balance between overall utility and protected group utility)
   - Select the rule with the highest score
   - Add the selected rule to the solution set
   - Update coverage for both protected and unprotected groups
6. If coverage thresholds are met, switch focus to improving utility for the protected group:
   - Select rules that maximize protected utility
   - Continue until maximum number of rules is reached or no improvement is possible
7. Calculate final metrics:
   - Expected utility
   - Protected expected utility
   - Overall coverage
   - Protected group coverage

## Output

Both algorithms will generate output files with their results:

- The greedy algorithm outputs to `experiment_results_greedy.csv`
- The CauSumX algorithm outputs to `experiment_results_causumx.csv`

These files contain detailed information about the selected rules, their fairness scores, and other relevant metrics.

