import torch
from selection.selectivesampler import SelectiveSampler


class HalfSampler(SelectiveSampler):
    """Example implementation of SelectiveSampler that only uses the first half of samples.

    This sampler demonstrates basic usage of SelectiveSampler by:
    1. Setting mask in pre_epoch to select first half of dataset
    2. Implementing required hook methods with simple pass-through behavior
    """

    def pre_epoch(self) -> None:
        """Set mask to select first half of samples before each epoch starts"""
        n = len(self.dataset)
        mask = [False] * n
        mask[n // 2 :] = [True] * (n - n // 2)
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
