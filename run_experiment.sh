# Download model 
tune download meta-llama/Llama-3.2-1B-Instruct --ignore-patterns "original/consolidated.00.pth"

# Begin finetuning with config
tune run recipe/full_finetune.py --config config/llama3_2/1b_full/train.yaml


# Evaluate base and finetuned model 
tune run recipe/eval.py --config config/llama3_2/1b_full/eval_base.yaml
tune run recipe/eval.py --config config/llama3_2/1b_full/eval_finetuned.yaml