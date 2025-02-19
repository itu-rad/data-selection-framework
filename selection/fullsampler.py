from selection.selectivesampler import SelectiveSampler


class FullSampler(SelectiveSampler):
    """Example implementation of SelectiveSampler that uses all of the samples."""

    def set_num_selected_samples(self):
        """Set expected number of selected samples for dataset len"""
        self._num_selected_samples = self.num_samples

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
