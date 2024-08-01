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

### Cosamix Algorithm

To run the Cosamix algorithm, use the following command:

```
python causumx.py
```

This will execute the Cosamix algorithm for causal inference and treatment effect estimation.

## File Structure and Main Functions

### greedy.py

This is the main algorithm file for the greedy approach. Key functions include:

- `main()`: Entry point of the program. Runs experiments for different values of k (number of rules).
- `run_experiment()`: Executes a single experiment for a specific k value.
- `greedy_fair_prescription_rules()`: Implements the greedy algorithm to select fair prescription rules.
- `calculate_fairness_score()`: Calculates the fairness score for a given rule.
- `score_rule()`: Assigns a score to a rule based on various factors including fairness and coverage.

### causumx.py

This file contains the implementation of the Cosamix algorithm. Key functions include:

- `main()`: Entry point for the Cosamix algorithm.
- `run_experiment()`: Executes the Cosamix experiment.
- `calculate_fairness_score()`: Calculates the fairness score for a given rule.
- `greedy_fair_prescription_rules()`: Implements the Cosamix version of the greedy algorithm.

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

## Algorithm Overview

1. Use Apriori algorithm to get grouping patterns.
2. For each grouping pattern:
   a. Find the treatment with the highest unfairness score.
   b. Calculate fairness score as: CATE / |CATE - CATE_p|
   c. If CATE = CATE_p, unfairness score = CATE
3. Use a greedy algorithm to choose the best rule (rule = grouping pattern + treatment pattern) based on:
   a. Highest CATE
   b. Highest unfairness score
   c. Highest coverage + protected coverage (marginal score)
4. Iterate K times to get K rules.
5. Calculate the final fairness constraint satisfaction.

## Output

Both algorithms will generate output files with their results:

- The greedy algorithm outputs to `experiment_results.csv`
- The Cosamix algorithm outputs to `experiment_results_not_male.csv`

These files contain detailed information about the selected rules, their fairness scores, and other relevant metrics.
