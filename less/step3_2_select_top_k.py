import argparse
import os
import torch
import yaml
import sys


from torchtune import config
from torchtune.datasets import ConcatDataset
from omegaconf import DictConfig, ListConfig, OmegaConf

 

def parse_cfg():
    argparser = argparse.ArgumentParser(
        description='Script for selecting the data for training')
    argparser.add_argument('--dataset_names', type=str,
                           ncfg='+', help='The path to the score file')
    argparser.add_argument('--datasets', type=str, ncfg='+',
                           help='The path of the training file that corresponds to the score file')
    argparser.add_argument('--target_task_name_names', type=str,
                           ncfg='+', help='The name of the target task')
    argparser.add_argument('--output_path', type=str,
                           default="selected_data", help='The path to the output')
    argparser.add_argument('--max_samples', type=int,
                           default=None, help='The maximum number of samples')
    argparser.add_argument('--percentage', type=float, default=None,
                           help='The percentage of the data to be selected')

    cfg = argparser.parse_cfg()

    return cfg

def get_dataset(cfg):
    if isinstance(cfg.datasets, ListConfig):
        # Test if dataset field in config with a single dataset is also considered a Listconfig
        datasets = [config.instantiate(current_dataset) for current_dataset in cfg.datasets]
        ds = ConcatDataset(datasets=datasets)
    
    else:
        ds = config.instantiate(cfg.datasets)
        
    return ds


def select_data_amount(cfg, total_samples):
    """
    This function determines the amount of data to be selected based on configuration.

    Args:
    cfg (DictConfig): The configuration dictionary.
    total_samples (int): The total number of samples in the dataset.

    Returns:
    data_amount_string (str): A string that indicates whether the data is selected based on percentage or max_samples.
    flag (bool): A boolean flag that indicates whether the data is selected based on percentage or max_samples.
    """

    # Determine selection based on percentage if provided
    if cfg.percentage is not None:
        top_k = int(cfg.percentage * total_samples)
        cfg.max_samples = top_k
        print(f'Percentage of selected data:{cfg.percentage}')
        print(f'top_k of data selected from percentage:{top_k}')
        return top_k

    # Default to max_samples if percentage is not provided
    else:
        top_k = int(cfg.max_samples)
        print(f'top_k of data selected via max_samples:{top_k}')
        return top_k
            
        

def setup(cfg, target_task_name):
    """
    This function sets up the configuration for the selection process.
    
    Args:
    cfg (DictConfig): The configuration dictionary.
    
    Returns:
    score_paths (list): A list of paths to the influence score files.
    num_samples (list): A list of the number of samples in each influence score file.
    data_amount_string (str): A string indicating the amount of data selected.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check that the lengths of dataset_names and datasets are equal
    n_datasets = len(cfg.dataset_names)
    assert n_datasets == len(cfg.datasets), "amount of datasets and amount of dataset names must be the same amount."
    
    # Check that either percentage or max_samples is not None
    assert cfg.percentage is not None or cfg.max_samples is not None, "Both 'percentage' and 'max_sample' config fields cannot be 'None'"
    
    # Initialize an empty list to hold the paths to the influence score files
    score_paths = []
    
    # Loop over each dataset name
    for dataset_name in cfg.dataset_names:
        # Create the path to the influence score file
        score_path = os.path.join(cfg.influence_score_path, target_task_name, f"{dataset_name}_influence_score.pt")
        # Add the path to the list
        score_paths.append(score_path)
    
    # Initialize an empty list to hold the number of samples in each influence score file
    num_samples = []
    total_samples = 0
    # Loop over each score path
    for score_path in score_paths:
        # Load the influence score file and get the number of samples
        current_file= torch.load(score_path, map_location=device)
        current_file_len= (len(current_file))
        num_samples.append(current_file_len)
        total_samples+=current_file_len
            
    # Select the amount of data to be selected
    top_k = select_data_amount(cfg, total_samples=total_samples)
    
    # Return the list of score paths, the list of number of samples, and the data amount string
    return score_paths, num_samples, top_k


def sort_scores(cfg, score_paths, num_samples, target_task_name): 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a tensor of influence scores from all datasets
    all_scores = []
    for score_path in score_paths:
        score = torch.load(score_path, map_location=device)
        all_scores.append(score)
    all_scores = torch.cat(all_scores, dim=0)

    # Create a tensor that keeps track of samples' index in their origin dataset
    file_specific_index = torch.cat([torch.arange(line_num) for line_num in num_samples]).to(device)
    
    # Create a tensor that keeps track of samples' origin dataset
    data_from = torch.cat([torch.ones(line_num, dtype=torch.long)* i for i, line_num in enumerate(num_samples)]).to(device)
    
    # Sort the scores and return how they were sorted 'sorted_index'
    sorted_scores, sorted_index = torch.sort(all_scores, dim=0, descending=True)
    
    # Use 'sorted_index' to apply the same sorting on the other tensors
    sorted_data_from = data_from[sorted_index]
    sorted_file_specific_index = file_specific_index[sorted_index]
        
    # Save the sorted scores in a csv
    sorted_score_file = os.path.join(cfg.output_path, target_task_name, f"sorted.csv")  
    if not os.path.exists(sorted_score_file):
        with open(sorted_score_file, 'w', encoding='utf-8') as file:
            file.write("file name, index, score\n")
            for score, index, name in zip(sorted_scores, sorted_file_specific_index, sorted_data_from):
                file.write(
                    f"{cfg.dataset_names[name.item()]}, {index.item()}, {round(score.item(), 6)}\n")
                
                
    return sorted_scores, sorted_file_specific_index, sorted_data_from

    #all_scores = 32,43,3,1,2,5,100
    #sorted_scores = 100,32,43,5,3,2,1
    #sorted_index =  6,0,1,7,12,5
    #file_specific_index = 0,1,2,3,4,5,6,0,1,2,3,4,5,0,1,2,3,4
    #data_from =0,0,0,0,0,0,0,1,1,

def select_and_write_samples(cfg, sorted_file_specific_index, sorted_data_from, num_samples, top_k, target_task_name):

    #
    all_lines = []
    for i, train_file in enumerate(cfg.datasets):
        with open(train_file, 'r', encoding='utf-8', errors='ignore') as file:
            all_lines.append(file.readlines()[:num_samples[i]])              
    
    # Extract top_k from both tensors
    final_index_list = sorted_file_specific_index[:top_k].tolist()
    final_data_from = sorted_data_from[:top_k].tolist()
    
    if cfg.percentage is not None:
        data_amount_name = f"p{cfg.percentage}"
    else:
        data_amount_name = f"num{cfg.max_samples}"
        
    with open(os.path.join(cfg.output_path, target_task_name, f"top_{data_amount_name}.jsonl"), 'w', encoding='utf-8', errors='ignore') as file:
        for index, data_from in zip(final_index_list, final_data_from):
            try:
                file.write(all_lines[data_from][index])
            except:
                import pdb
                pdb.set_trace()
    

def select_top_k(cfg:DictConfig="./data-selection-framework/less/config/llama3_2/step3.2_select_top_k.yaml") -> None:
    
    cfg = OmegaConf.load(cfg)
    
    # Loop over each target task name
    for target_task_name in cfg.target_task_names:
        score_paths, num_samples, top_k = setup(cfg)
        sorted_file_specific_index, sorted_data_from = sort_scores(cfg, score_paths, num_samples, target_task_name)
        select_and_write_samples(cfg, sorted_file_specific_index, sorted_data_from, num_samples, top_k, target_task_name)


if __name__ == "__main__":
    sys.exit(select_top_k())
