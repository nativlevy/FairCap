# Greedy Fair Prescription Rules Algorithm

This project implements a greedy algorithm for generating fair prescription rules. It aims to balance utility and fairness in decision-making processes, particularly focusing on protected groups.

## Slides

Experiment slides are available online at:
https://nativlevy.github.io/FairPrescriptionRules/slides.html

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

## Generating Slides

To generate the presentation slides, follow these steps:

1. Ensure you have run both the greedy and CauSumX algorithms, which will generate the necessary CSV files (`experiment_results_greedy.csv` and `experiment_results_causumx.csv`).

2. Run the slide generation script:
   ```
   python generate_slides.py
   ```

3. This will create a file named `slides.html` in the current directory.

The `generate_slides.py` script does the following:

1. Reads the CSV files containing the results from both algorithms.
2. Uses a Jinja2 template (`slides_template.html`) to create an HTML presentation.
3. For each value of k (4 to 7), it generates slides comparing the results of the greedy and CauSumX algorithms.
4. The generated slides include metrics such as execution time, expected utility, coverage, and the selected rules for each algorithm.

You can view the generated slides by opening the `slides.html` file in a web browser. The presentation uses reveal.js for a smooth slideshow experience.
