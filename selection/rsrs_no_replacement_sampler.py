from selection.selectivesampler import SelectiveSampler
from torch.utils.data import DistributedSampler


class Rsrs_No_Replacement_Sampler(SelectiveSampler):
    """Example implementation of RSRS (Repeated Sampling of Random Subsets) that uses all of the samples."""

    def set_num_selected_samples(self):
        """Set expected number of selected samples for dataset len"""
        # TODO: change percentage to an argument to easily try out different percentages
        # Should this consist of samples selected in each epoch or sum of all epochs?
        self._num_selected_samples = int(self.num_samples * 0.10)
    
    def __iter__(self):
        if self.epoch == 0:
            self.indices = list(DistributedSampler.__iter__(self))
        
        if self.mask is None:
            raise RuntimeError("No mask set - call set_mask() before iterating")

        # Corrected this to use i instead of idx to check the mask, otherwise it would not utilize shuffled indices correctly
        # indices = [idx for i, idx in enumerate(indices) if self.mask[idx]]
        selected_indices = [idx for i, idx in enumerate(self.indices) if self.mask[i]]
        if not selected_indices:
            raise RuntimeError("No samples selected - mask may be all False or unset")

        return iter(selected_indices)
        
    def pre_epoch(self,num_epochs) -> None:
        """Set mask to only select a random subset before each epoch starts"""
        dataset_size = len(self.dataset)
       
        # TODO: change percentage to an argument to easily try out different percentages
        subset_size = int(dataset_size / num_epochs)
        
        if (self.epoch == num_epochs-1): 
            start_idx = min(dataset_size -1, subset_size * self.epoch)
            # does not need to be dataset_size-1 because mask[start_idx:end_idx] is exclusive
            end_idx = dataset_size
        
        else : 
            start_idx = min(dataset_size -1, subset_size * self.epoch)
            end_idx = min(dataset_size-1, start_idx + subset_size)
            
         
        mask = [False] * dataset_size
        mask[start_idx:end_idx] = [True] * (end_idx - start_idx)
        
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

if __name__ == "__main__":
    ds = list(range(10))
    rsrs = Rsrs_No_Replacement_Sampler(ds,num_replicas=1, rank=0,shuffle=True, seed=0)
    epochs = 2
    for e in range(epochs):
        print(rsrs.dataset)
        rsrs.set_epoch(e)
        rsrs.pre_epoch(epochs)
        selected_indices = rsrs.__iter__()
        print(rsrs.indices)
        print(list(selected_indices))
        print(rsrs.mask)
        print("_____________")