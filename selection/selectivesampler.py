from abc import ABC, abstractmethod
from torch.utils.data import DistributedSampler


class SelectiveSampler(DistributedSampler, ABC):
    """A sampler that extends DistributedSampler to filter samples using a boolean mask.

    Inherits all functionality from DistributedSampler including:
    - Distributed sampling across multiple processes/GPUs
    - Shuffling capabilities
    - Deterministic seeding
    - Rank and num_replicas handling

    Adds the ability to:
    - Set a boolean mask to select which samples should be included in iteration
    - Automatically filter samples based on the mask during iteration
    - Validate mask length matches dataset length
    - Pre-epoch and post-epoch hooks for implementing custom logic

    Abstract methods that must be implemented:
    - pre_epoch(): Called before each epoch starts
    - post_epoch(): Called after each epoch ends
    - on_batch(idx, batch): Called before processing each batch with batch index and data
    - after_forward(idx, batch, current_loss): Called after forward pass with loss
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed
        )
        self.mask = None
        self.set_num_selected_samples()

    def set_mask(self, mask):
        """Set a boolean mask to filter samples"""
        if len(mask) != len(self.dataset):
            raise ValueError("Mask length must match dataset length")
        self.mask = mask

    def __iter__(self):
        indices = list(super().__iter__())

        if self.mask is None:
            raise RuntimeError("No mask set - call set_mask() before iterating")

        indices = [idx for i, idx in enumerate(indices) if self.mask[idx]]
        if not indices:
            raise RuntimeError("No samples selected - mask may be all False or unset")

        return iter(indices)

    def __len__(self) -> int:
        return self._num_selected_samples

    @abstractmethod
    def set_num_selected_samples(self) -> None:
        """Hook called on initialisation. Set expected number of selected samples for dataset len.
        Temporary workaround for __len__ method to return correct value. Must be implemented by subclasses
        """
        pass

    @abstractmethod
    def pre_epoch(self) -> None:
        """Hook called before each epoch starts. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def post_epoch(self) -> None:
        """Hook called after each epoch ends. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def on_batch(self, idx: int, batch: dict) -> None:
        """Hook called before processing each batch. Must be implemented by subclasses.

        Args:
            idx (int): The index/step number of the current batch
            batch (dict): The batch data dictionary containing inputs and labels
        """
        pass

    @abstractmethod
    def after_forward(self, idx: int, batch: dict, current_loss: float) -> None:
        """Hook called after model forward pass. Must be implemented by subclasses.

        Args:
            idx (int): The index/step number of the current batch
            batch (dict): The batch data dictionary containing inputs and labels
            current_loss (float): The loss value from the current forward pass
        """
        pass
