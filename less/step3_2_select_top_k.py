import json
import os
import torch
import sys
from torchtune import config
from omegaconf import DictConfig, ListConfig, OmegaConf
from datasets import load_dataset



def get_dataset(cfg):
    ds_list = []
    for current_dataset in cfg.dataset_paths:
        dataset_str= str(current_dataset)
        ds= load_dataset(path=dataset_str,split=cfg.split)
        ds_list.append(ds)

    return ds_list


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
        print(f'Percentage of selected data:{int(cfg.percentage*100)}%')
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
    
    # Create parent folder structure for top-k data files for each target task. 
    selected_data_path = os.path.join(cfg.output_path,target_task_name)
    os.makedirs(selected_data_path, exist_ok=True)  # Creates directory if it doesn't exist, doesn't overwrite if it does
    

    # Check that the lengths of dataset_names and datasets are equal
    assert len(cfg.dataset_paths) == len(cfg.dataset_names), "amount of datasets and amount of dataset names must be the same amount."
    
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
     #file_specific_index = 0,1,2,3,4,5,6,0,1,2,3,4,5,0,1,2,3,4
    file_specific_index = torch.cat([torch.arange(line_num) for line_num in num_samples]).to(device)
    
    # Create a tensor that keeps track of samples' origin dataset
    #data_from =0,0,0,0,0,0,0,1,1,
    data_from = torch.cat([torch.ones(line_num, dtype=torch.long)* i for i, line_num in enumerate(num_samples)]).to(device)
    
    
    #all_scores = 32,43,3,1,2,5,100
    print(f"all_scores shape:{all_scores}")
    print(f"file_specific_index shape:{file_specific_index}")
    print(f"data_from shape:{data_from}")

    
    # Sort the scores and return how they were sorted 'sorted_index'
    sorted_scores, sorted_index = torch.sort(all_scores, dim=0, descending=True)
    
    
    #sorted_index =  6,0,1,7,12,5
    print(f"sorted_index shape:{sorted_index}")
    
    #sorted_scores = 100,32,43,5,3,2,1
    print(f"sorted_index shape:{sorted_scores}")
    
    
    # Use 'sorted_index' to apply the same sorting on the other tensors
    sorted_data_from = data_from[sorted_index]
    print(f"sorted_data_from shape:{sorted_data_from}")
    
    sorted_file_specific_index = file_specific_index[sorted_index]
    print(f"sorted_file_specific_index shape:{sorted_file_specific_index}")
    
    
    # Save the sorted scores in a csv
    sorted_score_file = os.path.join(cfg.output_path, target_task_name, f"sorted.csv")  
    if not os.path.exists(sorted_score_file):
        with open(sorted_score_file, 'x', encoding='utf-8',) as file:
            file.write("file name, index, score\n")
            for score, index, name in zip(sorted_scores, sorted_file_specific_index, sorted_data_from):
                file.write(
                    f"{cfg.dataset_names[name.item()]}, {index.item()}, {round(score.item(), 6)}\n")
                
                
    return sorted_file_specific_index, sorted_data_from

  

def select_and_write_samples(cfg, sorted_file_specific_index, sorted_data_from, num_samples, top_k, target_task_name):

    datasets = get_dataset(cfg)
    
    # Extract top_k from both tensors
    final_index_list = sorted_file_specific_index[:top_k].tolist()
    final_data_from = sorted_data_from[:top_k].tolist()
    
    if cfg.percentage is not None:
        data_amount_name = f"p{cfg.percentage}"
    else:
        data_amount_name = f"num{cfg.max_samples}"
    
    top_k_path = os.path.join(cfg.output_path, target_task_name, f"top_{data_amount_name}.jsonl")    
    with open(top_k_path, 'w', encoding='utf-8', errors='ignore') as file:
        print(f"Writing top_k datasamples to {top_k_path}")
        for index, data_from in zip(final_index_list, final_data_from):    
            file.write(json.dumps(datasets[data_from][index]) + "\n")

         

def select_top_k(cfg:DictConfig="./less/config/llama3_2/step3_2_select_top_k.yaml") -> None:
    
    cfg = OmegaConf.load(cfg)
 
    # Loop over each target task name
    for target_task_name in cfg.target_task_names:
        score_paths, num_samples, top_k = setup(cfg,target_task_name)
        sorted_file_specific_index, sorted_data_from = sort_scores(cfg, score_paths, num_samples, target_task_name)
        select_and_write_samples(cfg, sorted_file_specific_index, sorted_data_from, num_samples, top_k, target_task_name)


if __name__ == "__main__":
    sys.exit(select_top_k())
