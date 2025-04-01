# Download model
model_company="meta-llama"
model_name="Llama-3.2-1B-Instruct"
 
echo "run_experiment.sh: Starting model download..."
tune download $model_company/$model_name --ignore-patterns "original/consolidated.00.pth" --output-dir ./model_cache/downloaded_models/$model_name
 
# Begin finetuning with config
echo "run_experiment.sh: Starting fine-tuning with config..."
 
# with radt:
python -u -m radt -e 137 --local --manual tune.py run recipe/test_full_finetune.py --config config/llama3_2/1b_full/train.yaml
 
# Evaluate base and finetuned model
echo "run_experiment.sh: Starting evaluation of base model..."
tune run recipe/eval.py --config config/llama3_2/1b_full/eval_base.yaml
tune run recipe/eval.py --config config/llama3_2/1b_full/eval_finetuned.yaml
