"""
LESS Data Selection Pipeline
==========================

Overview
--------

This Python script implements the LESS Data Selection Pipeline, a process for selecting 
the most informative data samples for fine-tuning a language model: https://github.com/princeton-nlp/LESS

The pipeline consists of several steps, each of which is implemented in a separate module.

Modules
-------
The pipeline consists of the following modules, which are different steps from the LESS code base, 
natively implemented in TorchTune via recipes:

* `step1_train_warmup_model`: Trains a warmup model on a small subset of the data.
* `step2_1_get_training_gradstore`: Computes the gradient store for the training data.
* `step2_2_get_validation_gradstore`: Computes the gradient store for the validation data.
* `step3_1_get_influence_scores`: Computes the influence scores for each data sample.
* `step3_2_select_top_k`: Selects the top-k data samples based on their influence scores.
* `step4_train_selected_data`: Trains the model on the selected data samples.


Each step utilizes a .yaml configuration file to specify the torchtune and custom parameters for the step.


Variables
---------

* `less_start_step`: The starting point of the pipeline. Can be set to any of the step numbers (1, 2.1, 2.2, 3.1, 3.2, 4).
* `less_stop_step`: The ending point of the pipeline. Can be set to any of the step numbers (1, 2.1, 2.2, 3.1, 3.2, 4).

Usage
-----

To run the pipeline, simply execute the script. The pipeline will start from the step specified by 
`less_start_step` and stop at the step specified by `less_stop_step`.

Notes
-----

* The pipeline assumes that the necessary dependencies, with appropriate versions are installed. 
* The dependency list can be found in the `Requirements.yaml` file. The dependencies are extensive, but 
  and can be installed using `conda env create -f Requirements.yaml`.
"""

import sys
from step1_train_warmup_model import train_warmup_model
from step2_1_get_training_gradstore import get_training_gradstore
from step2_2_get_validation_gradstore import get_validation_gradstore 
from step3_1_get_influence_scores import get_influence_scores
from step3_2_select_top_k import select_top_k
from step4_train_selected_data import train_selected_data


# define which LESS substep to begin with
less_start_step = 3.2
# define which LESS substep to stop at
less_stop_step = 4


# LESS Data Selection Pipeline: https://github.com/princeton-nlp/LESS
print("Starting LESS data selection pipeline")

# Step 1: Warmup training 
# Train model on 5% of data set, and cache model locally.
if less_start_step == 1 and less_stop_step >= 1:
    print("Starting step 1 of LESS: Training warmup model...")
    train_warmup_model()
    less_start_step = 2.1

# Step 2.1: Building the training gradient datastore.
if less_start_step == 2.1 and less_stop_step >= 2.1:
    print("Starting step 2.1 of LESS: Building training gradients datastore...")
    get_training_gradstore()
    less_start_step = 2.2  

# Step 2.2: Building the validation gradient datastore.
if less_start_step == 2.2 and less_stop_step >= 2.2:
    print("Starting step 2.2 of LESS: Building validation gradients datastore...")
    get_validation_gradstore()
    less_start_step = 3.1 
    
# Step 3.1: Computing influence scores.
if less_start_step == 3.1 and less_stop_step >= 3.1:
    print("Starting step 3.1 of LESS: Computing influence scores...")
    get_influence_scores()
    less_start_step = 3.2 
    
# Step 3.2: Selecting the top-k data samples and writing a new dataset locally.
if less_start_step == 3.2 and less_stop_step >= 3.2:
    print("Starting step 3.2 of LESS: Selecting and writing the top-k data samples...")
    select_top_k()
    less_start_step = 4 
    
# Step 4: Train with your selected data
if less_start_step == 4 and less_stop_step >= 4:
    print("doing step 4 of LESS") 
    train_selected_data()
    
sys.exit()