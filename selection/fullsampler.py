from selection.selectivesampler import SelectiveSampler


class FullSampler(SelectiveSampler):
    """Example implementation of SelectiveSampler that uses all of the samples."""

    def pre_epoch(self) -> None:
        """Set mask to select all samples before each epoch starts"""
        n = len(self.dataset)
        mask = [True] * n
        self.set_mask(mask)

    def on_scoring_phase(self) -> None:
        pass

    def score(self, recipe, idx, batch):
        pass

    def on_training_phase(self) -> None:
        pass

    def on_batch(self, idx: int, batch: dict) -> None:
        pass

    def after_forward(self, idx: int, batch: dict, current_loss: float) -> None:
        pass

    def post_epoch(self) -> None:
        pass
