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

def setup(cfg):
    
    n_datasets = len(cfg.dataset_names)
    assert len(n_datasets) == len(cfg.datasets),"dataset_names and datasets must have the same length"
    assert cfg.percentage is not None or cfg.max_samples is not None,"Both 'percentage' and 'max_sample' config fields cannot be 'None'"
    
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

   
    for target_task_name in cfg.target_task_names:
        output_path = os.path.join(cfg.output_path, target_task_name)

        score_paths = [os.path.join(
            # task_name variable should perhaps be train_file_name
            output_path, f"{task_name}_influence_score.pt") for task_name in cfg.train_file_names]
        num_samples = []
        for score_path in score_paths:
            num_samples.append(
                len(torch.load(score_path, map_location=device)))
        
        
        total_samples = sum(num_samples)
        if cfg.percentage is not None:
            cfg.max_samples = int(cfg.percentage * total_samples)
            data_amount_name = f"p{cfg.percentage}"
        else:
            data_amount_name = f"num{cfg.max_samples}"

        return score_path, num_samples 
    



# We might want to change somes lines, such that either percentage or max_samples is supplied. 
# Will also need to be reflected in select_data.sh. 
# Will make the code easier to read and simplier to use. 
def recipe_main(cfg:DictConfig="./data-selection-framework/less/config/llama3_2/step3.2_selest_top_k.yaml") -> None:
    
    cfg = OmegaConf.load(cfg)
    ds = get_dataset(cfg)


if __name__ == "__main__":
    sys.exit(recipe_main())
    
    

    assert len(cfg.train_file_names) == len(cfg.datasets)
    assert cfg.percentage is not None or cfg.max_samples is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_datasets = len(cfg.train_file_names)

    # target_task_name variable should perhaps be target_task_name_name
    for target_task_name in cfg.target_task_name_names:
        output_path = os.path.join(cfg.output_path, target_task_name)

        score_paths = [os.path.join(
            # task_name variable should perhaps be train_file_name
            output_path, f"{task_name}_influence_score.pt") for task_name in cfg.train_file_names]
        num_samples = []
        for score_path in score_paths:
            num_samples.append(
                len(torch.load(score_path, map_location=device)))
        
        total_samples = sum(num_samples)
        if cfg.percentage is not None:
            cfg.max_samples = int(cfg.percentage * total_samples)
            data_amount_name = f"p{cfg.percentage}"
        else:
            data_amount_name = f"num{cfg.max_samples}"

        return score_path, num_samples






        all_scores = []
        # train_file and cfg.datasets is never referenced in the loop.
        # Should perhaps be deleted. 
        for score_path, train_file in zip(score_paths, cfg.datasets):
            score = torch.load(score_path, map_location=device)
            all_scores.append(score)
        all_scores = torch.cat(all_scores, dim=0)
        
        

        # sort the scores and output the corresponding data index
        file_specific_index = torch.cat(
            [torch.arange(line_num) for line_num in num_samples]).to(device)
        data_from = torch.cat([torch.ones(line_num, dtype=torch.long)
                              * i for i, line_num in enumerate(num_samples)]).to(device)
        sorted_scores, sorted_index = torch.sort(
            all_scores, dim=0, descending=True)
        sorted_score_file = os.path.join(output_path, f"sorted.csv")

    #all_scores = 32,43,3,1,2,5,100
    #sorted_scores = 100,32,43,5,3,2,1
    #sorted_index =  6,0,1,7,12,5
    #file_specific_index = 0,1,2,3,4,5,6,0,1,2,3,4,5,0,1,2,3,4
    #data_from =0,0,0,0,0,0,0,1,1,

        # revisit later
        data_from = data_from[sorted_index]
        sorted_index = file_specific_index[sorted_index]
        
        # This part might have been utilized for as postprocessing step of making plots. 
        if not os.path.exists(sorted_score_file):
            with open(sorted_score_file, 'w', encoding='utf-8') as file:
                file.write("file name, index, score\n")
                for score, index, name in zip(sorted_scores, sorted_index, data_from):
                    file.write(
                        f"{cfg.train_file_names[name.item()]}, {index.item()}, {round(score.item(), 6)}\n")

        # topk_scores and topk_indices is never used.
        topk_scores, topk_indices = torch.topk(
            all_scores.float(), cfg.max_samples, dim=0, largest=True)

        all_lines = []
        for i, train_file in enumerate(cfg.datasets):
            with open(train_file, 'r', encoding='utf-8', errors='ignore') as file:
                # maybe [:num_samples[i]] slice is redundant since file.readlines()
                all_lines.append(file.readlines()[:num_samples[i]])
        
                               
        
        final_index_list = sorted_index[:cfg.max_samples].tolist()
        final_data_from = data_from[:cfg.max_samples].tolist()
        with open(os.path.join(output_path, f"top_{data_amount_name}.jsonl"), 'w', encoding='utf-8', errors='ignore') as file:
            for index, data_from in zip(final_index_list, final_data_from):
                try:
                    file.write(all_lines[data_from][index])
                except:
                    import pdb
                    pdb.set_trace()
