from selection.selectivesampler import SelectiveSampler
import subprocess
import os
import sys
from less.step1_warmup import warmup

class LESSBasedSampler(SelectiveSampler):
    """Sampler which utilizes LESS data selection method.
       This sampler can only be utilized with LORA """

    """ LESS utilizes 5% of data samples for warmup training. This can be change for experimentation"""

    def __init__(self, dataset, start_from_step, stop_after_step, num_replicas=None, rank=None, shuffle=True, seed=0):
      super().__init__(
          dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed
      )
      self.less_step = start_from_step or 1
      self.less_stop = stop_after_step or 4
      self.mask = None
      self.no_grad_scoring = False
    
    def pre_epoch(self) -> None:
        """Set mask to select all samples before each epoch starts"""
        
        # Use to test: python tune.py run less/recipe/test_full_finetune.py --config less/config/llama3_2/1b_full/train_less.yaml
        
        # LESS Data Selection Pipeline: https://github.com/princeton-nlp/LESS
        print("Starting LESS data selection pipeline")
        # Step 1: Warmup training 
        # Train model on 5% of data set, and cache model locally.
        if self.less_step == 1 and self.less_stop >= 1:
            print("doing step 1 of LESS")
            warmup()
            self.less_step += 1 
        
        # Step 2: Building the gradient datastore
        if self.less_step == 2 and self.less_stop >= 2:
            print("doing step 2 of LESS")
            self.less_step += 1  
        
        # Step 3: Selecting data for a task
        if self.less_step == 3 and self.less_stop >= 3:
            print("doing step 3 of LESS")
            self.less_step += 1  

        # Step 4: Train with your selected data
        if self.less_step == 4 and self.less_stop >= 4:
            print("doing step 4 of LESS") 
            
        sys.exit()
        
    def post_epoch(self) -> None:
        """No-op post epoch hook"""
        pass

    def on_batch(self, idx: int, batch: dict) -> None:
        """No-op batch hook"""
        pass

    def after_forward(self, idx: int, batch: dict, current_loss: float) -> None:
        """No-op after forward hook"""
        pass
    
    def inform_logits(self, idx: int, batch: dict, current_loss: float) -> None:
        """Hook called after model forward pass. Must be implemented by subclasses.

        Args:
            idx (int): The index/step number of the current batch
            batch (dict): The batch data dictionary containing inputs and labels
            current_loss (float): The loss value from the current forward pass
        """
        pass

    def sample(self) -> None:
        """Called after first phase forward pass in sample-then-batch
        """
        pass
