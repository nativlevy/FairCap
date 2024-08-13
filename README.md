# Greedy Fair Prescription Rules Algorithm

This project implements a greedy algorithm for generating fair prescription rules in machine learning models. It aims to balance utility and fairness in decision-making processes, particularly focusing on protected groups.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/greedy-fair-prescription-rules.git
   cd greedy-fair-prescription-rules
   ```

2. Install Graphviz:
   ```
   brew install graphviz
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

## Data

The main dataset used in this project is `so_countries_col_new.csv`, located in the `data/` directory. This dataset contains information about developers, including demographic data and salary information.

## Running the Algorithms

### Greedy Algorithm

To run the greedy algorithm, execute the following command in your terminal:

```
python greedy.py
```

This will run the main experiment using the greedy approach for fair prescription rules.

### CauSumX Algorithm

To run the CauSumX algorithm, use the following command:

```
python CauSumX.py
```

This will execute the CauSumX algorithm for causal inference and treatment effect estimation.

## File Structure and Main Functions

### greedy.py

This is the main algorithm file for the greedy approach. Key functions include:

- `main()`: Entry point of the program. Runs experiments for different values of k (number of rules).
- `run_experiment()`: Executes a single experiment for a specific k value.
- `greedy_fair_prescription_rules()`: Implements the greedy algorithm to select fair prescription rules.
- `calculate_fairness_score()`: Calculates the fairness score for a given rule.
- `score_rule()`: Assigns a score to a rule based on various factors including fairness and coverage.

### CauSumX.py

This file contains the implementation of the CauSumX algorithm. Key functions include:

- `main()`: Entry point for the CauSumX algorithm.
- `run_experiment()`: Executes the CauSumX experiment.
- `calculate_fairness_score()`: Calculates the fairness score for a given rule.
- `greedy_fair_prescription_rules()`: Implements the CauSumX version of the greedy algorithm.

### Algorithms.py

Contains helper functions for the main algorithms. Key functions include:

- `getAllGroups()`: Generates all possible grouping patterns using the Apriori algorithm.
- `getGroupstreatmentsforGreeedy()`: Gets treatments for each group using a greedy approach.
- `getHighTreatments()`: Finds the best treatment for a given group that maximizes fairness and effectiveness.

### Utils.py

Provides utility functions used throughout the project. Important functions include:

- `estimateATE()`: Estimates the Average Treatment Effect.
- `getTreatmentCATE()`: Calculates the Conditional Average Treatment Effect for a given treatment.

### Data2Transactions.py

Handles data preprocessing and transformation. Key functions:

- `removeHeader()`: Processes the dataframe for transaction analysis.
- `getRules()`: Generates association rules using the Apriori algorithm.

### dags.py

Contains the definition of the causal graph (SO_DAG) used in the project.

## Greedy Algorithm Overview

1. Use Apriori algorithm to get grouping patterns.
2.

The scoring function for a rule could be: 
score = (utility_protected + utility_unprotected) * unfairness_score * coverage_factor

Where:
- coverage_factor rewards rules that cover more individuals, especially from the protected group
- unfairness_score is calculated as described in step 2

Example of choosing the best treatment pattern for a given grouping pattern:
p1: CATE = 100, CATE_p (protected) = 50 -> score = 100/|100 - 50| = 2
p2: CATE = 80, CATE_p = 70 -> score = 80/|80 - 70| = 8
p3: CATE = 30, CATE_p = 60 -> score = |30/|30 - 60|| = 1
p4: CATE = 80, CATE_p = 80 -> score = 80 (when CATE = CATE_p, score = CATE)

The best treatment pattern is p4 and the second best is p2.

## Output

Both algorithms will generate output files with their results:

- The greedy algorithm outputs to `experiment_results_greedy.csv`
- The CauSumX algorithm outputs to `experiment_results_causumx.csv`

These files contain detailed information about the selected rules, their fairness scores, and other relevant metrics.

## Slides

https://nativlevy.github.io/FairPrescriptionRules/slides.html
