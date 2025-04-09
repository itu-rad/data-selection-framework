from selection.selectivesampler import SelectiveSampler
import subprocess
import os
import sys
from less.step1_warmup import warmup

class LESSBasedSampler(SelectiveSampler):
    """Sampler which utilizes LESS data selection method.
       This sampler can only be utilized with LORA """

    """ LESS utilizes 5% of data samples for warmup training. This can be change for experimentation"""

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
      super().__init__(
          dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed
      )
      self.mask = None
      self.no_grad_scoring = False
    
    def pre_epoch(self) -> None:
        """Set mask to select all samples before each epoch starts"""
        
        # STEP 1: train model on 5% of data set, and cache model locally.   
        warmup()
        
        # STEP 2:
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
