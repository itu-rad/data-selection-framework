
import os
import sys
import csv

from omegaconf import DictConfig, OmegaConf
import torch

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
        
        # returned changed config 
        return cfg 
    
    
def compute_influence_scores(cfg):
    # calculate the influence score for each training dataset for each validation task
    for train_file_name in cfg.train_file_names:
        for validation_task in cfg.validation_task_name :
            influence_scores = 0
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
                num_gradients = validation_info.shape[0]
                print(f"NOTICE num_gradients: {num_gradients}")
               
    
                if not torch.is_tensor(validation_info):
                    validation_info = torch.tensor(validation_info)
                validation_info = validation_info.to(device).float()
    
    
                
                influence_scores += cfg.checkpoint_avg_lr[i] * calculate_influence_score(
                training_info=training_info, validation_info=validation_info)
                
                
            print(f"Shape of influence_scores:{influence_scores.shape}")    
            # Step 1: Reshape
            reshaped = influence_scores.reshape(influence_scores.shape[0], num_gradients[validation_task], -1)
            print("After reshape:", reshaped.shape)  # Shape: [A, B, C]

            # Step 2: Mean over last dim
            meaned = reshaped.mean(dim=-1)
            print("After mean:", meaned.shape)  # Shape: [A, B]

            # Step 3: Max over last dim
            maxed = meaned.max(dim=-1)  # Returns a namedtuple (values, indices)
            print("After max:", maxed)
            print("Max values:", maxed.values.shape)  # Shape: [A]
            print("Max indices:", maxed.indices.shape)  # Shape: [A]

            # Step 4: Take only the values (index 0 of the namedtuple)
            influence_scores = maxed.values
            print("Final influence_score:", influence_scores.shape)

            output_dir = os.path.join(cfg.output_dir, validation_task)
            os.makedirs(output_dir, exist_ok=True)  # Creates output_dir safely, even if it already exists

            output_file = os.path.join(output_dir, f"{train_file_name}_influence_scores.pt")
            print(f"Saving influence score to {output_file}")
            torch.save(influence_scores, output_file)



def get_avg_lr_csv(cfg): 
    avg_lr_path = cfg.avg_lr_path
    
    with open(avg_lr_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        learning_rates = [float(row['Average_Learning_Rate']) for row in reader]
    
    checkpoint_avg_lr = learning_rates
    cfg.checkpoint_avg_lr = checkpoint_avg_lr
    # return changed cfg
    
    return cfg 
def get_influence_scores(cfg: DictConfig = "less/config/llama3_2/step3.1_get_influence_scores.yaml"):
    
    cfg = OmegaConf.load(cfg)
    changed_config1 = get_avg_lr_csv(cfg)
    changed_config2 =  renormalize_avg_lr(changed_config1)
    compute_influence_scores(changed_config2)
    
 
if __name__ == "__main__":
    sys.exit(get_influence_scores())
