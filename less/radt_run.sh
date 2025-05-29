# Run experiment with radt:

# LORA:
python -u -m radt -e 137 --local --manual tune.py run less/experiment_recipes/test_lora_finetune.py --config less/config/llama3_2/1b_lora/train.yaml

# FULL:
# python -u -m radt -e 137 --local --manual tune.py run less/experiment_recipes/test_full_finetune.py --config less/config/llama3_2/1b_full/train.yaml


# Testing full LESS in radt for resource consumption:
python -u -m radt -e 137 --local --manual tune.py run less/step1_train_warmup_model.py --config less/config/llama3_2/step1_train_warmup_model.yaml

python -u -m radt -e 137 --local --manual tune.py run less/step2_1_get_training_gradstore.py --config less/config/llama3_2/step2_1_get_training_gradstore.yaml

python -u -m radt -e 137 --local --manual tune.py run less/step2_2_get_validation_gradstore.py --config less/config/llama3_2/step2_2_get_validation_gradstore.yaml

python -u -m radt -e 137 --local --manual tune.py run less/step3_1_get_influence_scores.py --config less/config/llama3_2/step3_1_get_influence_scores.yaml

python -u -m radt -e 137 --local --manual tune.py run less/step3_2_select_top_k.py --config less/config/llama3_2/step3_2_select_top_k.yaml

python -u -m radt -e 137 --local --manual tune.py run less/step4_train_selected_data.py --config less/config/llama3_2/step4_train_selected_data.yaml