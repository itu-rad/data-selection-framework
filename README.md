# Data Selection Framework
This is a suite based on [torchtune](https://github.com/pytorch/torchtune) that aims to fairly compare a wide range of data selection methods, providing a overview of the field and introducing resource metrics via [radT](https://github.com/Resource-Aware-Data-systems-RAD/radt).
We evaluate data selection methods on a range of tuning tasks.

&nbsp;

## Getting Started

```bash
# Create the 'selection' conda environment
conda env create -f conda.yaml
conda activate selection



# If you want to install additional dependencies add dependencies in conda.yaml and run:
conda env update --file conda.yaml



# Now set the HF_TOKEN environment variable in your conda environment
conda env config vars set HF_TOKEN=<enter token here>

```

Follow the instructions on the official [`meta-llama`](https://huggingface.co/meta-llama) repository to ensure you have access to the official Llama model weights. Once you have confirmed access, you can run the following command to download the weights to your local machine. This will also download the tokenizer model and a responsible use guide.


torchtune supports the following models:

| Model                                         | Sizes     |
|-----------------------------------------------|-----------|
| [Llama3.3](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_3)    | 70B [[models](torchtune/models/llama3_3/_model_builders.py), [configs](recipes/configs/llama3_3/)]        |
| [Llama3.2-Vision](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2#-llama-3.2-vision-models-(11b/90b)-)    | 11B, 90B [[models](torchtune/models/llama3_2_vision/_model_builders.py), [configs](recipes/configs/llama3_2_vision/)]        |
| [Llama3.2](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2)    | 1B, 3B [[models](torchtune/models/llama3_2/_model_builders.py), [configs](recipes/configs/llama3_2/)]        |
| [Llama3.1](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1)    | 8B, 70B, 405B [[models](torchtune/models/llama3_1/_model_builders.py), [configs](recipes/configs/llama3_1/)]        |
| [Llama3](https://llama.meta.com/llama3)    | 8B, 70B [[models](torchtune/models/llama3/_model_builders.py), [configs](recipes/configs/llama3/)]        |
| [Llama2](https://llama.meta.com/llama2/)   | 7B, 13B, 70B [[models](torchtune/models/llama2/_model_builders.py), [configs](recipes/configs/llama2/)]        |
| [Code-Llama2](https://ai.meta.com/blog/code-llama-large-language-model-coding/)   | 7B, 13B, 70B [[models](torchtune/models/code_llama2/_model_builders.py), [configs](recipes/configs/code_llama2/)] |
| [Mistral](https://huggingface.co/mistralai)   | 7B [[models](torchtune/models/mistral/_model_builders.py), [configs](recipes/configs/mistral/)] |
| [Gemma](https://huggingface.co/collections/google/gemma-release-65d5efbccdbb8c4202ec078b)   | 2B, 7B [[models](torchtune/models/gemma/_model_builders.py), [configs](recipes/configs/gemma/)] |
| [Gemma2](https://huggingface.co/docs/transformers/main/en/model_doc/gemma2)   | 2B, 9B, 27B [[models](torchtune/models/gemma2/_model_builders.py), [configs](recipes/configs/gemma2/)] |
| [Microsoft Phi3](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3) | Mini [[models](torchtune/models/phi3/), [configs](recipes/configs/phi3/)]
| [Qwen2](https://qwenlm.github.io/blog/qwen2/) | 0.5B, 1.5B, 7B [[models](torchtune/models/qwen2/), [configs](recipes/configs/qwen2/)]
| [Qwen2.5](https://qwenlm.github.io/blog/qwen2.5/) | 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B [[models](torchtune/models/qwen2_5/), [configs](recipes/configs/qwen2_5/)]

We recommend getting started with the small [Llama3.2](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2) models.

&nbsp;

### Downloading the model  

```bash
# Insert huggingface model company and model name from huggingface model page.
model_company="meta-llama"
model_name="Llama-3.2-1B-Instruct" 

tune download $model_company/$model_name --ignore-patterns "original/consolidated.00.pth" --output-dir ./model_cache/downloaded_models/$model_name
```

&nbsp;

### Creating recipes and configs

To list all available torchtune recipes & configs

```bash
tune ls
```

## Creating a recipe at the path
```bash

recipe="full_finetune_single_device"
recipe_path="./recipe/full_finetune"
tune cp $recipe $recipe_path --make-parents
```

## Creating a config at the path. 
By default configs will utilize linux 'tmp' folder. This will result in downloaded and finetuned models being deleted after each session. 

```bash 
# The current local model_cache pathsystem needs to be integrated as part of the config download pipeline. 
# TO BE IMPLEMENTED
model_config="llama3_2/1B_full_single_device"
config_path="./config/llama3_2/1b_full/train.yaml"
tune cp $model_config $config_path --make-parents
```

&nbsp;

### Running finetuning recipes

You can finetune Llama3.2 1B on a single GPU using the following command (with/without radT):

```bash
python tune.py run recipe/full_finetune.py --config config/llama3_2/1b_full/train.yaml
python -m radt --local --manual tune.py run recipe/full_finetune.py --config config/llama3_2/1b_full/train.yaml
```
 
Or with LoRA:
 
```bash
python tune.py run recipe/lora_finetune.py --config config/llama3_2/1b_lora/train.yaml
python -m radt --local --manual tune.py run recipe/lora_finetune.py --config config/llama3_2/1b_lora/train.yaml
```

Saving with RadT with a specific MLflow experiment ID

```bash
# set experiment_id to the MLflow experiment ID 
experiment_id=" " 
# full model 
python -m radt -e $experiment_id --local --manual tune.py run recipe/full_finetune.py --config config/llama3_2/1b_full/train.yaml
```

```bash
# With lora model
python -m radt -e $experiment_id --local --manual tune.py run recipe/lora_finetune.py --config config/llama3_2/1b_full/train.yaml
```


&nbsp;

### Evaluating Models

Training configs should be accompanied with evaluation configs. To evaluate the models trained above:

```bash
tune run recipe/eval.py --config config/llama3_2/1b_full/eval_base.yaml
tune run recipe/eval.py --config config/llama3_2/1b_full/eval_finetuned.yaml
```

Or with LoRA:

```bash
tune run recipe/eval.py --config config/llama3_2/1b_lora/eval_base.yaml
tune run recipe/eval.py --config config/llama3_2/1b_lora/eval_finetuned.yaml
```

&nbsp;

## Evaluation tasks
A full list of evaluation tasks can be found here: [https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/README.md](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/README.md)
Additionally, a full list of datasets to train on can be found here: [https://pytorch.org/torchtune/0.2/api_ref_datasets.html#datasets](https://pytorch.org/torchtune/0.2/api_ref_datasets.html#datasets).

Further torchtune examples: [https://github.com/pytorch/torchtune/blob/main/docs/source/tutorials/llama3.rst](https://github.com/pytorch/torchtune/blob/main/docs/source/tutorials/llama3.rst)


&nbsp;

### Infering Models 

Begin by creating a custom generation config, either by running the following commands 
or creating your own: 

``` bash
tune cp generation ./custom_generation_config.yaml 

``` 


Infering the model by: changeing the "user" field value within the config and running the following command:
```bash 
tune run generate --config ./custom_generation_config.yam
```


Infering the model by: using torch tune cli run the following command:
```bash 
tune run generate --config ./custom_generation_config.yaml prompt.user=<Your Prompt Here> 
```
