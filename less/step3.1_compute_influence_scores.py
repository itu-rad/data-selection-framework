import argparse
import os
import sys
import csv
import yaml


from omegaconf import DictConfig, OmegaConf
import torch

argparser = argparse.ArgumentParser(
    description='Script for selecting the data for training')
argparser.add_argument('--gradient_path', type=str, default="{} ckpt{}",
                       help='The path to the gradient file')
argparser.add_argument('--train_file_names', type=str, nargs='+',
                       help='The name of the training file')
argparser.add_argument('--ckpts', type=int, nargs='+',
                       help="Checkpoint numbers.")
argparser.add_argument('--checkpoint_weights', type=float, nargs='+',
                       help="checkpoint weights")
argparser.add_argument('--target_task_names', type=str,
                       nargs='+', help="The name of the target tasks")
argparser.add_argument('--validation_gradient_path', type=str,
                       default="{} ckpt{}", help='The path to the validation gradient file')
argparser.add_argument('--output_path', type=str, default="selected_data",
                       help='The path to the output')


args = argparser.parse_args()

N_SUBTASKS = {"mmlu": 57, "bbh": 27, "tydiqa": 9,"truthful_qa":1}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_influence_score(training_info: torch.Tensor, validation_info: torch.Tensor):
    """Calculate the influence score.

    Args:
        training_info (torch.Tensor): training info (gradients/representations) stored in a tensor of shape N x N_DIM
        validation_info (torch.Tensor): validation info (gradients/representations) stored in a tensor of shape N_VALID x N_DIM
    """
    # This operation computes a similarity, correlation, or influence between training and validation data
    # N x N_VALID
    influence_scores = torch.matmul(
        training_info, validation_info.transpose(0, 1))
    return influence_scores

def renormalize_avg_lr(cfg) -> None:
    # renormalize the checkpoint weights
    if sum(cfg.checkpoint_avg_lr) != 1:
        s = sum(cfg.checkpoint_avg_lr)
        cfg.checkpoint_avg_lr = [i/s for i in cfg.checkpoint_avg_lr]
    
    
    
def indtilvidere(cfg):
    # calculate the influence score for each training dataset for each validation task
    for train_file_name in cfg.train_file_names:
        for validation_task in cfg.validation_task_name :
            influence_score = 0
            for i, ckpt in enumerate(cfg.checkpoints):
                
                # load training gradients 
                training_path = os.path.join(cfg.training_gradient_path,train_file_name,f"epoch_{ckpt}",f"dim{cfg.gradient_projection_dimension}","all_orig.pt")
                training_info = torch.load(training_path)
                    
                if not torch.is_tensor(training_info):
                    training_info = torch.tensor(training_info)
                training_info = training_info.to(device).float()
    
    
                # load validation gradients
                validation_path = os.path.join(cfg.validation_gradient_path,validation_task,f"epoch_{ckpt}",f"dim{cfg.gradient_projection_dimension}","all_orig.pt")
                validation_info = torch.load(validation_path)
    
                if not torch.is_tensor(validation_info):
                    validation_info = torch.tensor(validation_info)
                validation_info = validation_info.to(device).float()
    
    
                
                influence_score += cfg.checkpoint_avg_lr[i] * calculate_influence_score(
                training_info=training_info, validation_info=validation_info)
                
            influence_score = influence_score.reshape(influence_score.shape[0], N_SUBTASKS[target_task_name], -1).mean(-1).max(-1)[0]
            
            
            
            output_dir = os.path.join(cfg.output_dir, target_task_name)
            
            output_file = os.path.join(cfg.output_dir, target_task_name, f"{train_file_name}_influence_score.pt")
            torch.save(influence_score, output_file)
            print(f"Saved influence score to {output_file}")



def get_avg_lr_csv(cfg): 
    avg_lr_path = cfg.avg_lr_path
    
    with open(avg_lr_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        learning_rates = [float(row['Average_Learning_Rate']) for row in reader]
    
    checkpoint_avg_lr = learning_rates
    cfg.checkpoint_avg_lr = checkpoint_avg_lr
    
def recipe_main(cfg: DictConfig = "less/config/llama3_2/step3.1_compute_influence_scores.yaml"):
    
    cfg = OmegaConf.load(cfg)
    indtilvidere(cfg)
    get_avg_lr_csv(cfg)
 
    
    
    
if __name__ == "__main__":
    sys.exit(recipe_main())
