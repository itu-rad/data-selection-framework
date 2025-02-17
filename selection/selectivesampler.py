from torch.utils.data import DistributedSampler


class SelectiveSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed
        )
        self.mask = None

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
