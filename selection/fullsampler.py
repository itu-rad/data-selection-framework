from selection.selectivesampler import SelectiveSampler


class FullSampler(SelectiveSampler):
    """Example implementation of SelectiveSampler that uses all of the samples."""

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