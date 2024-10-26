# Greedy Fair Prescription Rules Algorithm
In this project, we implement a 3-step algorithms that generate prescriptions(rules) to increase/decrease the value of an attribute. Meanwhile, we can protecting a specified group by setting a minimum coverage of each rule, or that of a rule set. We can also proctect this group by capping the gap between the difference of benefit protected and non-protected group get. The algorithm can be broken down into 3 steps: group mining, treatment mining and rule selection. The details can be found in the [paper]()


## Setup <a name="setup"></a>
1. Clone this repository:
```
git clone https://github.com/USERNAME/FairPrescriptionRules
cd FairPrescriptionRules
```
You can run the algorithm either locally or **remotely(recommended)**
### Local Setup (Skip to Remote setup if you wish to run the experiment on cloudlab)
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
#### After installing dependencies. Run the sanity test to verify the result
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
Under the `FairPrescriptionRules/output/Local\ Sanity\ Test/greedy` directory, the following files can be found:
- selected_rules.json: rules generated from treatment mining
- mined_rules.json: rules selected by greedy algorithm
- experiment_results_greedy.csv: information inlcluding expected utilty, coverage and fairness during the selection process

### Remote Setup on [CloudLab](https://docs.cloudlab.us/getting-started.html)
Step 0. Get a [CloudLab](https://docs.cloudlab.us/getting-started.html) account for free
Step 1. Set up remote servers 
1. Instiantiate an experiment (https://www.cloudlab.us/instantiate.php)
2. Select the `causal` profile     
   <img width="600" alt="image" src="https://github.com/user-attachments/assets/ea7b7dd6-4d03-4bdb-9364-34042bf1339b">
3. Specify the number of nodes needed. For example, to run 9 variants of FariCap in parallel, you need 9 nodes.
   Then Select the `urn:publicid:IDN+utah.cloudlab.us+image+fair-prescrip-PG0:causal` image, which is an Ubuntu image that has all the required packages installed      
   <img width="600" alt="image" src="https://github.com/user-attachments/assets/3fccb68b-0e52-4d78-b717-3cfc093d9674">      
   Then select the machine. We recommend xl170 in Utah cluster. You can also use m510 or cl6525-25g machines in Utah   
   <img width="600" alt="image" src="https://github.com/user-attachments/assets/f8a9f906-64c3-4ace-a6fc-c7b2c932740a">        
   Then choose an experiment name. For simplicy, use `remote` as the experiment name as shown in the example configuration. You need to change the configuration accordingly.      
   <img width="600" alt="image" src="https://github.com/user-attachments/assets/d1076d6f-87e6-4a6f-8f4f-caa5605197a2">     
   Finally, specify the length of your experiment (e.g. 1 hour is sufficient for running the full Stackoverflow dataset with constraints)    
   <img width="600" alt="image" src="https://github.com/user-attachments/assets/28aaa261-7aa4-4c2d-9bab-d167597d3e85">     
   When all the nodes are ready, experiments are ready to run       
   <img width="600" alt="image" src="https://github.com/user-attachments/assets/91f106d5-597c-4bde-8bae-9264bec85465">   

4. (Optional). Run the sanity test to verify the result
```
cd reproducibility
sh remote_sanity_check.sh
```
Expected output:
```
BEGIN
logs.tar                                                                    100%   10KB  35.0KB/s   00:00    
0
DONE
```
In the `FairPrescriptionRules/output/greedy` directory, the following files can be found: 
- `experiment_results_greedy.csv`
- `mined_rules.json`
- `selected_rules.json`
- `stderr.log`: warnings and errors occured on remote server
- `stdout.log`: output including execution time break down, number of groupings and number of mined treatments



## Data & Experiment Specification
### Data
The datasets we use for evaluations are `German credit` and `StackOverflow`. They can be found in the `data` directory. 

To run the experiment, we need both **data configuration** and **experiment configuration**.
### A **data configuration** is json file that contains the following field
1. `_dataset_path`: home path to the dataset and data configurations
2. `_datatable_path`: local path to the dataframe (in .csv format)
3. `_dag_path`: path to the causal directed acyclic graph (in .dot format)
4. `_immutable_attributes`, `_mutable_attributes`: immutbale and mutable (mutally exclusive to immutbale) attributes
5. `_protected_attributes`, `_protected_values`: protected group, i.e. the individuals whose protected_attributes = protected_values
6. `_target_outcome`, targeted attributes that the algorithm aims to maxize
### A **experiment configuration** is json file that contains the following field
1. `_is_remote`: flag that indicates whether the experiment runs remotely
2. `_expmt_title`: experiment tile
3. `_cloudlab_user`, `_cloudlab_postfix`: cloudlab configuration
4. `_cloudlab_nodes`: array of node names on cloudlab
5. `_models`: array of models that contains model name, starting script and variants if there is any

### A model contains the following fields
1. `_name` (required): name of the model, used in the output directory
2. `_start` (required): starting script 
3. `_coverage_constraint` (optional): contains name of the variant (group or rule), threshold, and protected threshold
3. `_fairness_constraint` (optional): contains name of the variant (group_sp, individual_sp, grouo_bgl, individual_bgl) and threshold

See examples of configuration [here](https://github.com/USERNAME/FairPrescriptionRules/tree/master/experiment-scripts/experiment-configs/sanity)
## Running the Algorithms

Locate to the script directory
```
cd FairPrescriptionRules/experiment-scripts
```
Then run `run_experiment.py` in the following semantics:  
```
python run_experiment.py PATH_TO_DATA_CONFIGURATION PATH_TO_EXPERIMENT_CONFIGURATION
``` 
See the section above regarding the specification of data configuration and experiment configuration
### To reproduce the result from the paper
```
cd FairPrescriptionRules/reproducibility
```
And execute the corresponding script
e.g. `sh stackoverflow/so_full.sh`


## Output
Output can be found in the output path specified in the configuration. For each model, the following file will be generated
- `experiment_results_greedy.csv`: information including expected utility, fairness, coverage, during the rule section
- `mined_rules.json`: rules generated for each grouping pattern
- `selected_rules.json`: selected rules from the previous step
- `stderr.log`: warnings and errors occured on remote server
- `stdout.log`: output including execution time break down, number of groupings and number of mined treatments

## Result replicaiton 
We recommend using CloudLab remote servers. The following guide is based on remote experiment
To run all 9 possible fairness and coverage variants on stackoverflow:
```
cd FairPrescriptionRules/experiment-scripts/reproducibility
sh stackoverflow/so_full.sh
```
You can also customize 
- Thresholds and variants in `FairPrescriptionRules/experiment-scripts/experiment-configs/so/remote_full.json`
- Mutable, immutbale, target attributes, protected attributes or protected value in `FairPrescriptionRules/data/stackoverflow/config_full.json`
- DAG defined in `FairPrescriptionRules/data/stackoverflow/so.dot`
