import math
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

    def __init__(
        self,
        dataset,
        batch_size=1,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        sampling_ratio: float = 0.5,
        num_passes=-1,
    ):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed
        )
        self.mask = None
        self.batch_size = batch_size
        self.no_grad_scoring = False

        self.sampling = False
        self.sampling_indices = None
        self.sampling_ratio = sampling_ratio
        self.num_passes = num_passes
        if self.num_passes == -1:
            self.num_passes = math.ceil(
                len(dataset) / self.batch_size * self.sampling_ratio
            )
        elif self.num_passes < 1:
            self.num_passes = math.ceil(1 / self.num_passes)

    def set_mask(self, mask):
        """Set a boolean mask to filter samples"""
        if len(mask) != len(self.dataset):
            raise ValueError("Mask length must match dataset length")
        self.mask = mask

    def __iter__(self):
        # Override the __iter__ method to filter samples based on the mask
        # Keep in mind that the RNG comes from the data loader shuffling indices, not the mask itself
        # This means that the mask is applied after the shuffling
        # Keeping the same mask between epochs does not guarantee same data elements
        if self.sampling:
            return iter(self.sampling_indices[self.sampling_start : self.sampling_end])

        indices = self.get_iterator()

        if self.mask is None:
            raise RuntimeError("No mask set - call set_mask() before iterating")

        indices = [idx for idx in indices if self.mask[idx]]

        if not indices:
            raise RuntimeError("No samples selected - mask may be all False or unset")

        return iter(indices)

    def __len__(self) -> int:
        if self.sampling:
            return self.sampling_end - self.sampling_start

        if self.mask is not None:
            return sum(self.mask)
        return len(self.dataset)

    def _prepare_epoch(self):
        self.sampling_indices = list(self.get_iterator())

    def _prepare_scoring_phase(self, selection_pass):
        # Reset the loss buffer and mask at the start of each scoring phase.
        n = len(self.dataset)
        mask = [True] * n
        self.sampling = True
        self.sampling_start = selection_pass * n // self.num_passes
        self.sampling_end = (selection_pass + 1) * n // self.num_passes

        self.set_mask(mask)

    def _prepare_training_phase(self):
        # Disable sampling mode
        self.sampling = False

    def get_iterator(self):
        """Get the iterator for the dataset. This is used to get the indices for the sampler."""
        return super().__iter__()

    @abstractmethod
    def pre_epoch(self) -> None:
        """Hook called before each epoch starts. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def on_scoring_phase(self) -> None:
        """Hook called before each sampling phase. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def score(self, recipe, idx, batch):
        pass

    @abstractmethod
    def on_training_phase(self) -> None:
        """Hook called before each training phase. Must be implemented by subclasses."""
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

    @abstractmethod
    def post_epoch(self) -> None:
        """Hook called after each epoch ends. Must be implemented by subclasses."""
        pass
