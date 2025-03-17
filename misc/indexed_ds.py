from torch.utils.data import Dataset

class IndexedWrapperDataset(Dataset):
    """Generic dataset wrapper that adds sample indices to the items."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        item = self.dataset[index]
        assert isinstance(item, dict)
        assert "sample_id" not in item or item["sample_id"] == index
        item['sample_id'] = index
        return item

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, attr):
        """
        Forward any attribute or method access not defined here to the underlying dataset.
        This includes any additional method calls.
        """
        if attr == "dataset": # Prevent recursive call when dataset attribute isn't yet set
            raise AttributeError("Attribute 'dataset' is not defined.")
        return getattr(self.dataset, attr)
