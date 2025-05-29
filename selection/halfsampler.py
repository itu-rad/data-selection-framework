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
        mask = [True] * n
        mask[n // 2 :] = [False] * (n - n // 2)
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

    def inform_logits(self, sample_ids: list[int], logits: torch.Tensor, shifted_labels: torch.Tensor) -> None:
        pass

    def sample(self) -> None:
        pass
    
    
if __name__ == "__main__":
    ds = list(range(10))
    pbs = HalfSampler(ds,num_replicas=1)
    epochs = 1
    for e in range(epochs):
        print(pbs.dataset)
        pbs.set_epoch(e)
        pbs.pre_epoch()
        selected_indices = pbs.__iter__()
        print(list(selected_indices))
        print(pbs.mask)
        print("_____________")