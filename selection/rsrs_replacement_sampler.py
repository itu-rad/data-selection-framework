from selection.selectivesampler import SelectiveSampler


class Rsrs_Replacement_Sampler(SelectiveSampler):
    """Example implementation of RSRS (Repeated Sampling of Random Subsets) that uses all of the samples."""

    def set_num_selected_samples(self):
        """Set expected number of selected samples for dataset len"""
        # TODO: change percentage to an argument to easily try out different percentages
        # Should this consist of samples selected in each epoch or sum of all epochs?
        self._num_selected_samples = int(self.num_samples * 0.10)

    def pre_epoch(self) -> None:
        """Set mask to only select a random subset before each epoch starts"""
        dataset_size = len(self.dataset)
        # TODO: change percentage to an argument to easily try out different percentages
        subset_size = int(0.10 * dataset_size)  
        
        mask = [False] * dataset_size
        mask[:subset_size] = [True] * subset_size
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
    ds = list(range(20))
    rsrs = Rsrs_Replacement_Sampler(ds,num_replicas=1, rank=0,shuffle=True, seed=0)
    epochs = 10
    for e in range(epochs):
        print(rsrs.dataset)
        rsrs.set_epoch(e)
        rsrs.pre_epoch()
        shuffled_ds = rsrs.__iter__()
        print(list(shuffled_ds))
        print(rsrs.mask)
        print("_____________")