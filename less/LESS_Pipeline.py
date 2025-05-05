import sys
from less.step1_train_warmup_model import train_warmup_model
from less.step2_1_get_training_gradstore import get_training_gradstore
from less.step2_2_get_validation_gradstore import get_validation_gradstore 
from less.step3_1_get_influence_scores import get_influence_scores
from less.step3_2_select_top_k import select_top_k
from less.step4_train_selected_data import train_selected_data


# define which LESS substep to begin with
less_step = 1
# define which LESS substep to stop at
less_stop = 4
"""Set mask to select all samples before each epoch starts"""

# LESS Data Selection Pipeline: https://github.com/princeton-nlp/LESS
print("Starting LESS data selection pipeline")

# Step 1: Warmup training 
# Train model on 5% of data set, and cache model locally.
if less_step == 1 and less_stop >= 1:
    print("Starting step 1 of LESS: Training warmup model...")
    train_warmup_model()
    less_step = 2.1

# Step 2.1: Building the training gradient datastore.
if less_step == 2.1 and less_stop >= 2.1:
    print("Starting step 2.1 of LESS: Building training gradients datastore...")
    get_training_gradstore()
    less_step = 2.2  

# Step 2.2: Building the validation gradient datastore.
if less_step == 2.2 and less_stop >= 2.2:
    print("Starting step 2.2 of LESS: Building validation gradients datastore...")
    get_validation_gradstore()
    less_step = 3.1 
    
# Step 3.1: Computing influence scores.
if less_step == 3.1 and less_stop >= 3.1:
    print("Starting step 3.1 of LESS: Computing influence scores...")
    get_influence_scores()
    less_step = 3.2 
    
# Step 3.2: Selecting the top-k data samples and writing a new dataset locally.
if less_step == 3.2 and less_stop >= 3.2:
    print("Starting step 3.2 of LESS: Selecting and writing the top-k data samples...")
    select_top_k()
    less_step = 4 
# Step 4: Train with your selected data
if less_step == 4 and less_stop >= 4:
    print("doing step 4 of LESS") 
    train_selected_data()
    
sys.exit()