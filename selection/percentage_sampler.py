from selection.selectivesampler import SelectiveSampler


class PercentageBasedSampler(SelectiveSampler):
    """Example implementation of SelectiveSampler that uses all of the samples."""

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, percentage=0.05):
       super().__init__(
           dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed
       )
       self.percentage = percentage
       self.mask = None
       self.no_grad_scoring = False
    
    def pre_epoch(self) -> None:
        """Set mask to select all samples before each epoch starts"""
        n = len(self.dataset)
        sample_amount = int(n * self.percentage)
        mask = [False] * n 
        mask[:sample_amount] = [True] * sample_amount
        print(f"Dataset:{len(ds)}")
        print(f"Mask length:{len(mask)}")
        
        self.set_mask(mask)
        

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

    
    


if __name__ == "__main__":
    ds = list(range(100))
    pbs = PercentageBasedSampler(ds,rank=0,num_replicas=1)
    epochs = 2
    for current_epoch in range(epochs):
        print(pbs.dataset)
        pbs.set_epoch(current_epoch)
        pbs.pre_epoch()
        selected_indices = pbs.__iter__()
        print(f"dataset indicies{list(selected_indices)}")
        print (f"Epoch nr:{current_epoch}")
        print(f"Mask: {pbs.mask}")
        print("_____________")