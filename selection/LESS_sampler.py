from selection.selectivesampler import SelectiveSampler
import subprocess
import os
import sys
from less.step1_train_warmup_model import train_warmup_model
from less.step2_1_get_training_gradstore import get_training_gradstore
from less.step2_2_get_validation_gradstore import get_validation_gradstore 
from less.step3_1_get_influence_scores import get_influence_scores
from less.step3_2_select_top_k import select_top_k

class LESSBasedSampler(SelectiveSampler):
    """Sampler which utilizes LESS data selection method.
       This sampler can only be utilized with LORA """

    """ LESS utilizes 5% of data samples for warmup training. This can be changed for experimentation"""

    def __init__(self, dataset, start_from_step, stop_after_step, num_replicas=None, rank=None, shuffle=True, seed=0):
      super().__init__(
          dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed
      )
      self.less_step = start_from_step or 1
      self.less_stop = stop_after_step or 4
      self.mask = None
    
    def pre_epoch(self) -> None:
        """Set mask to select all samples before each epoch starts"""
        
        # LESS Data Selection Pipeline: https://github.com/princeton-nlp/LESS
        print("Starting LESS data selection pipeline")
        
        # Step 1: Warmup training 
        # Train model on 5% of data set, and cache model locally.
        if self.less_step == 1 and self.less_stop >= 1:
            print("Starting step 1 of LESS: Training warmup model...")
            train_warmup_model()
            self.less_step = 2.1
        
        # Step 2.1: Building the training gradient datastore.
        if self.less_step == 2.1 and self.less_stop >= 2.1:
            print("Starting step 2.1 of LESS: Building training gradients datastore...")
            get_training_gradstore()
            self.less_step = 2.2  
        
        # Step 2.2: Building the validation gradient datastore.
        if self.less_step == 2.2 and self.less_stop >= 2.2:
            print("Starting step 3.1 of LESS: Building validation gradients datastore...")
            get_validation_gradstore()
            self.less_step = 3.1 
            
        # Step 3.1: Computing influence scores.
        if self.less_step == 3.1 and self.less_stop >= 3.1:
            print("Starting step 3.1 of LESS: Computing influence scores...")
            get_influence_scores()
            self.less_step = 3.2 
            
        # Step 3.2: Selecting the top-k data samples and writing a new dataset locally.
        if self.less_step == 3.2 and self.less_stop >= 3.2:
            print("Starting step 3.2 of LESS: Selecting and writing the top-k data samples...")
            select_top_k()
            self.less_step = 4 

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
