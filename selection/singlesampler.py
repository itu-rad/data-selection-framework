from selection.selectivesampler import SelectiveSampler
from torch.utils.data import DistributedSampler


class SingleSampler(SelectiveSampler):
    """Example implementation of SelectiveSampler that uses all of the samples."""

    def set_num_selected_samples(self):
        """Set expected number of selected samples for dataset len"""
        self._num_selected_samples = self.num_samples

    def __iter__(self):
        self.indices_permutation = list(DistributedSampler.__iter__(self))
        
        self.indices = [self.indices_permutation[0]] * len(self.indices_permutation)
        
        if self.mask is None:
            raise RuntimeError("No mask set - call set_mask() before iterating")

        selected_indices = [idx for i, idx in enumerate(self.indices) if self.mask[i]]
        if not selected_indices:
            raise RuntimeError("No samples selected - mask may be all False or unset")

        return iter(selected_indices)
    
    def pre_epoch(self) -> None:
        """Set mask to select all samples before each epoch starts"""
        n = len(self.dataset)
        mask = [True] * n
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
    ds = list(range(12))
    singleSampler = SingleSampler(ds,num_replicas=1, rank=0,shuffle=True, seed=0)
    epochs = 4
    for e in range(epochs):
        print(singleSampler.dataset)
        singleSampler.set_epoch(e)
        singleSampler.pre_epoch()
        selected_indices = singleSampler.__iter__()
        print(singleSampler.indices_permutation)
        print(list(selected_indices))
        print(singleSampler.mask)
        print("_____________")